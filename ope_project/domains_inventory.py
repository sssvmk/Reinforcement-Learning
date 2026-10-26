"""Inventory domain on one M5 series: vectorised environment, features, behavior policy, agents and ground truth.
Ground truth = Monte Carlo rollouts of the epsilon-soft target policy on the REAL demand path (common random numbers
across agents). Costs are assumptions (price, unit cost 0.6*mean price, holding 0.5% of unit cost, stockout 0.2*price)."""
from __future__ import annotations
import numpy as np
from common import Log, new_window_ids
from agents.base import Agent
from agents.rules_inventory import build_inventory_rule_agents
from agents.learned import build_learned_agents

Q_MULT = np.array([0, 0.5, 1, 2, 3, 4, 5, 7.0])


class InvDomain:
    name = "inventory"
    n_actions = 8

    def __init__(self, series, fold, cfg, lead_time=None, tag=""):
        self.s, self.fold, self.cfg = series, fold, cfg
        ic = cfg["inventory"]
        self.gamma, self.H = cfg["mdp"]["gamma"], cfg["mdp"]["H"]
        self.L = int(ic["lead_time"] if lead_time is None else lead_time)
        lo, hi = fold.train
        d = series.demand
        self.mean_d = float(d[lo:hi].mean())
        self.u = max(self.mean_d, 0.3)
        pr = series.price.copy()
        self.mean_price = float(np.nanmean(pr[lo:hi]))
        self.price = np.where(np.isfinite(pr), pr, self.mean_price)
        self.uc = ic["unit_cost_frac"] * self.mean_price
        self.hold = ic["holding_frac"] * self.uc
        self.pen_frac = ic["stockout_frac"]
        self.fix = ic["fixed_cost_frac"] * self.uc
        self.reward_scale = 1.0 / (self.mean_price * self.u)
        self.cost_bps = 0.0
        self.tag = f"{series.sid}_L{self.L}_g{self.gamma}" + tag
        # order-quantity levels (strictly increasing integers)
        Q = np.round(self.u * Q_MULT)
        for i in range(1, len(Q)):
            Q[i] = max(Q[i], Q[i - 1] + 1)
        self.Q = Q
        self.init_on = float(np.ceil(2 * self.u))
        self.npipe = max(self.L - 1, 0)
        self.state_step_max = self.H // 2
        self._feat_tables()
        self._J, self._B = {}, {}
        self.behavior_agent = None

    # ---------------------------------------------------------------- ranges
    def rng_of(self, kind):
        f = self.fold
        return {"train": f.train, "ope": f.ope, "tune": f.tune, "val": f.val, "test": f.test}[kind]

    # ---------------------------------------------------------------- dynamics (vectorised)
    def vec_step(self, t, on, pipe, q):
        on_now = on + (q if self.L == 0 else 0.0)
        dem = self.s.demand[t]
        sales = np.minimum(on_now, dem)
        lost = dem - sales
        end = on_now - sales
        pr = self.price[t]
        r = pr * sales - self.uc * q - self.hold * end - self.pen_frac * pr * lost - self.fix * (q > 0)
        if self.L == 0:
            arrive, pipe2 = np.zeros_like(on), pipe
        elif self.L == 1:
            arrive, pipe2 = q, pipe
        else:
            arrive, pipe2 = pipe[:, 0], np.concatenate([pipe[:, 1:], q[:, None]], axis=1)
        return r, t + 1, end + arrive, pipe2, sales, lost

    def raw(self, t, on, pipe):
        return np.concatenate([t[:, None].astype(float), on[:, None], pipe], axis=1)

    # ---------------------------------------------------------------- features
    def _feat_tables(self):
        s, n = self.s, self.s.N
        lo, hi = self.fold.train
        d = s.demand
        cs = np.concatenate([[0], np.cumsum(d)])
        idx = np.arange(n)
        a = np.maximum(idx - 7, 0)
        cnt = np.maximum(idx - a, 1)
        trail = (cs[idx] - cs[a]) / cnt / self.u                      # past 7 days only
        self.f_trail = np.digitize(trail, [0.5, 0.8, 1.2, 1.6])       # 5
        self.f_price = np.digitize(self.price / self.mean_price, [0.9, 0.97, 1.03, 1.1])  # 5
        self.f_wd, self.f_ev, self.f_snap = s.weekday.astype(int), s.event.astype(int), s.snap.astype(int)

    def _groups(self, raw, dense):
        t = raw[:, 0].astype(int)
        on = raw[:, 1]
        pp = raw[:, 2:].sum(axis=1) if raw.shape[1] > 2 else np.zeros(len(raw))
        u = self.u
        ib = np.digitize((on + pp) / u, [0.5, 1, 2, 3, 4, 6, 8])       # 8
        ob = np.digitize(on / u, [0.01, 0.5, 1, 2, 3, 5])              # 7
        pb = np.digitize(pp / u, [0.01, 1, 2, 4])                      # 5
        wd, ev, sn, pr, tr = self.f_wd[t], self.f_ev[t], self.f_snap[t], self.f_price[t], self.f_trail[t]
        g = [(7, wd), (2, ev), (2, sn), (5, pr), (5, tr), (8, ib), (7, ob), (5, pb), (56, ib * 7 + ob),
             (40, ib * 5 + tr), (40, ib * 5 + pr), (56, ib * 7 + wd), (1, np.zeros(len(t), int))]
        if dense:
            g += [(35, ob * 5 + tr), (280, (ib * 7 + ob) * 5 + tr), (35, wd * 5 + pr), (35, wd * 5 + tr),
                  (40, ib * 5 + pb), (16, (ib * 2 + ev) * 1)]
        return g

    def _phi(self, raw, dense):
        g = self._groups(np.asarray(raw, float), dense)
        d = sum(s for s, _ in g)
        X = np.zeros((len(raw), d), dtype=np.float32)
        off, rows = 0, np.arange(len(raw))
        for size, idx in g:
            X[rows, off + idx] = 1.0
            off += size
        return X / np.sqrt(len(g))

    def phi(self, raw):
        return self._phi(raw, False)

    def phi_dense(self, raw):
        return self._phi(raw, True)

    @property
    def dim(self):
        return self.phi(self.raw(np.zeros(1, int), np.zeros(1), np.zeros((1, self.npipe)))).shape[1]

    # ---------------------------------------------------------------- behavior and logs
    def _ensure_behavior(self):
        if self.behavior_agent is None:
            from agents.rules_inventory import tune_behavior_base_stock
            self.behavior_agent = tune_behavior_base_stock(self)
        return self.behavior_agent

    def _starts_ok(self, kind):
        lo, hi = self.rng_of(kind)
        look = self.H + (self.H // 2 if kind in ("ope", "test") else 0)
        cand = np.arange(lo, hi - look)
        ok = np.array([self.s.valid[c:c + look + 1].all() for c in cand], dtype=bool)
        return cand[ok]

    def sample_log(self, kind, n_windows, eps, rng):
        cand = self._starts_ok(kind)
        if len(cand) == 0:
            raise ValueError("split too short for a window")
        starts = np.sort(rng.choice(cand, size=n_windows, replace=n_windows > len(cand)))
        return self.simulate_log(starts, eps, rng, kind)

    def simulate_log(self, starts, eps, rng, kind=""):
        ba = self._ensure_behavior()
        W, H, nA = len(starts), self.H, self.n_actions
        t = starts.astype(int).copy()
        on = np.full(W, self.init_on)
        pipe = np.zeros((W, self.npipe))
        raw = np.zeros((W, H + 1, 2 + self.npipe))
        act = np.zeros((W, H), dtype=np.int64)
        r = np.zeros((W, H))
        bp = np.zeros((W, H))
        for k in range(H):
            raw[:, k] = self.raw(t, on, pipe)
            rule = ba.act(raw[:, k])
            a = np.where(rng.random(W) < eps, rng.integers(0, nA, W), rule)
            bp[:, k] = (1 - eps) * (a == rule) + eps / nA
            act[:, k] = a
            r[:, k], t, on, pipe, _, _ = self.vec_step(t, on, pipe, self.Q[a])
        raw[:, H] = self.raw(t, on, pipe)
        return Log(raw, act, r, bp, new_window_ids(W), starts, kind, eps)

    # ---------------------------------------------------------------- rollouts and truth
    def rollout(self, act_fn, eps, t0, on0, pipe0, n_steps, rng):
        """Discounted return of epsilon-soft policy act_fn from each (t0,on0,pipe0). Uses the real demand path."""
        E = len(t0)
        t, on, pipe = t0.astype(int).copy(), on0.copy(), pipe0.copy()
        U = rng.random((n_steps, E))
        Ra = rng.integers(0, self.n_actions, (n_steps, E))
        G, disc = np.zeros(E), 1.0
        for k in range(n_steps):
            g = act_fn(self.raw(t, on, pipe))
            a = np.where(U[k] < eps, Ra[k], g)
            r, t, on, pipe, _, _ = self.vec_step(t, on, pipe, self.Q[a])
            G += disc * r
            disc *= self.gamma
        return G

    def _day_values(self, act_fn, eps, kind, nmc=24):
        cand = self._starts_ok(kind)
        E = len(cand) * nmc
        t0 = np.repeat(cand, nmc)
        rng = np.random.default_rng(900 + len(cand))      # common random numbers across agents
        G = self.rollout(act_fn, eps, t0, np.full(E, self.init_on), np.zeros((E, self.npipe)), self.H, rng)
        return dict(zip(cand.tolist(), G.reshape(len(cand), nmc).mean(axis=1)))

    def truth(self, agents, log, eps_t, kind, state_sample, nmc_state=12):
        starts = log.start.astype(int)
        Jw = np.zeros((len(agents), log.W))
        for i, ag in enumerate(agents):
            key = (ag.name, eps_t, kind)
            if key not in self._J:
                self._J[key] = self._day_values(ag.act, eps_t, kind)
            tab = self._J[key]
            Jw[i] = [tab[int(s)] for s in starts]
        st = log.raw[state_sample[:, 0], state_sample[:, 1]]
        t0, on0, p0 = np.repeat(st[:, 0], nmc_state), np.repeat(st[:, 1], nmc_state), np.repeat(st[:, 2:], nmc_state, axis=0)
        vs = np.zeros((len(agents), len(st)))
        for i, ag in enumerate(agents):
            rng = np.random.default_rng(555)
            vs[i] = self.rollout(ag.act, eps_t, t0, on0, p0, self.H, rng).reshape(len(st), nmc_state).mean(axis=1)
        return Jw, vs

    def behavior_value(self, log, eps_b, kind):
        key = (eps_b, kind)
        if key not in self._B:
            self._B[key] = self._day_values(self._ensure_behavior().act, eps_b, kind)
        return float(np.mean([self._B[key][int(s)] for s in log.start]))

    # ---------------------------------------------------------------- deployment performance (test period, eps = 0)
    def test_perf(self, agent):
        lo, hi = self.rng_of("test")
        t, on, pipe = np.array([lo]), np.array([self.init_on]), np.zeros((1, self.npipe))
        tot = dem = sold = 0.0
        for _ in range(hi - lo):
            a = agent.act(self.raw(t, on, pipe))
            r, t, on, pipe, sales, lost = self.vec_step(t, on, pipe, self.Q[a])
            tot += float(r[0]); dem += float(sales[0] + lost[0]); sold += float(sales[0])
        n = hi - lo
        return dict(cum_return=tot / (n * self.mean_price * self.u), sharpe=np.nan, fill_rate=sold / max(dem, 1e-9))

    # ---------------------------------------------------------------- agent pool
    def build_pool(self, seed, use_torch=True, cache_dir=None):
        agents = build_inventory_rule_agents(self)
        if use_torch:
            tr = self.sample_log("train", 600, self.cfg["behavior"]["eps_default"], np.random.default_rng(seed + 1))
            agents += build_learned_agents(self, tr, self.gamma, seed, lengths=(3000, 10000, 20000))
        self.pool = agents
        return agents
