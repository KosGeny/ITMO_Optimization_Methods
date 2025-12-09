using System;
using System.Collections.Generic;
using System.Linq;

class Program
{
    struct State
    {
        public int c1, c2, d, f;
        public State(int c1, int c2, int d, int f)
        {
            this.c1 = c1; this.c2 = c2; this.d = d; this.f = f;
        }
        public override int GetHashCode() => (c1, c2, d, f).GetHashCode();
        public override bool Equals(object obj) => obj is State s &&
            s.c1 == c1 && s.c2 == c2 && s.d == d && s.f == f;
    }

    struct Action
    {
        public int x1, x2, x3;
        public Action(int a, int b, int c)
        { x1 = a; x2 = b; x3 = c; }
    }


    static Dictionary<int, Dictionary<string, (double p, double k1, double k2, double kd)>> stageData =
        new Dictionary<int, Dictionary<string, (double, double, double, double)>>()
    {
        [1] = new Dictionary<string, (double, double, double, double)> {
            ["good"] = (0.6,1.2,1.1,1.07),
            ["neutral"] = (0.3,1.05,1.02,1.03),
            ["bad"] = (0.1,0.8,0.95,1.0)
        },
        [2] = new Dictionary<string, (double, double, double, double)> {
            ["good"] = (0.3,1.4,1.15,1.01),
            ["neutral"] = (0.2,1.05,1.0,1.0),
            ["bad"] = (0.5,0.6,0.9,1.0)
        },
        [3] = new Dictionary<string, (double, double, double, double)> {
            ["good"] = (0.4,1.15,1.12,1.05),
            ["neutral"] = (0.4,1.05,1.01,1.01),
            ["bad"] = (0.2,0.7,0.94,1.0)
        }
    };

    static Dictionary<string, double> comm = new Dictionary<string, double> {
        ["c1_buy"]=0.04, ["c1_sell"]=0.04,
        ["c2_buy"]=0.07, ["c2_sell"]=0.07,
        ["d_buy"]=0.05, ["d_sell"]=0.05
    };

    static Dictionary<string,int> pkg = new Dictionary<string,int> {
        ["c1"]=25, ["c2"]=200, ["d"]=100
    };

    static Dictionary<(int, State), double> memo = new Dictionary<(int, State), double>();
    static Dictionary<(int, State), Action> policy = new Dictionary<(int, State), Action>();


    static int Q(int v, int step) =>
        (int)(Math.Round(v / (double)step) * step);

    static State Quantize(State s) =>
        new State(
            Q(s.c1, 25),
            Q(s.c2, 200),
            Q(s.d, 100),
            Q(s.f, 25)
        );


    static void Main()
{
    State start = new State(100, 800, 400, 600);

    double maxIncome = W(1, start);
    Console.WriteLine($"Максимальный ожидаемый доход W1(S1) = {maxIncome:F2} д.е.");

    State cur = start;
    for (int stage = 1; stage <= 3; stage++)
    {
        Action a = policy[(stage, cur)];
        Console.WriteLine($"Этап {stage}: x1={a.x1}, x2={a.x2}, x3={a.x3}");

        cur = ExpectedTransition(stage, cur, a);
        cur = Quantize(cur);
    }
}

    static State ExpectedTransition(int stage, State s, Action a)
{
    double kc1 = 0, kc2 = 0, kd = 0;

    foreach(var kv in stageData[stage])
    {
        kc1 += kv.Value.p * kv.Value.k1;
        kc2 += kv.Value.p * kv.Value.k2;
        kd  += kv.Value.p * kv.Value.kd;
    }

    int nc1 = (int)Math.Round((s.c1 + pkg["c1"] * a.x1) * kc1);
    int nc2 = (int)Math.Round((s.c2 + pkg["c2"] * a.x2) * kc2);
    int nd  = (int)Math.Round((s.d  + pkg["d"] * a.x3) * kd);
    int nf  = (int)Math.Round(s.f - ActionCost(a.x1,a.x2,a.x3));

    return new State(nc1, nc2, nd, nf);
}

    static double W(int stage, State s)
    {
        if (stage == 4)
            return s.c1 + s.c2 + s.d + s.f;

        State qs = Quantize(s);

        if (memo.ContainsKey((stage, qs)))
            return memo[(stage, qs)];

        double best = double.NegativeInfinity;
        Action bestAction = new Action(0, 0, 0);

        foreach (var a in GetActions(qs))
        {
            double ev = 0;

            foreach (var kv in stageData[stage])
            {
                var next = Quantize(Transition(stage, qs, a, kv.Key));
                ev += kv.Value.p * W(stage + 1, next);
            }

            if (ev > best)
            {
                best = ev;
                bestAction = a;
            }
        }
        Console.WriteLine(
            $"[STAGE {stage}] Best action for state (c1={qs.c1}, c2={qs.c2}, d={qs.d}, f={qs.f}) → " +
            $"x1={bestAction.x1}, x2={bestAction.x2}, x3={bestAction.x3}, EV={best:F2}");

        memo[(stage, qs)] = best;
        policy[(stage, qs)] = bestAction;
        return best;
    }

    static List<Action> GetActions(State s)
    {
        List<Action> res = new List<Action>();

        int min_x1 = (int)Math.Ceiling((30.0 - s.c1) / pkg["c1"]);
        int max_x1 = (int)Math.Floor(s.f / (pkg["c1"] * (1 + comm["c1_buy"])));

        int min_x2 = (int)Math.Ceiling((150.0 - s.c2) / pkg["c2"]);
        int max_x2 = (int)Math.Floor(s.f / (pkg["c2"] * (1 + comm["c2_buy"])));

        int min_x3 = (int)Math.Ceiling((100.0 - s.d) / pkg["d"]);
        int max_x3 = (int)Math.Floor(s.f / (pkg["d"] * (1 + comm["d_buy"])));

        for (int x1 = min_x1; x1 <= max_x1; x1++)
        for (int x2 = min_x2; x2 <= max_x2; x2++)
        for (int x3 = min_x3; x3 <= max_x3; x3++)
            if (IsValid(s, x1, x2, x3))
                res.Add(new Action(x1, x2, x3));

        return res;
    }

    static bool IsValid(State s, int x1, int x2, int x3)
    {
        if (s.c1 + pkg["c1"] * x1 < 30) return false;
        if (s.c2 + pkg["c2"] * x2 < 150) return false;
        if (s.d + pkg["d"] * x3 < 100) return false;

        double cost = ActionCost(x1, x2, x3);
        if (s.f - cost < 0) return false;

        return true;
    }

    static double ActionCost(int x1, int x2, int x3)
    {
        double cost = 0;

        if (x1 > 0) cost += pkg["c1"] * x1 * (1 + comm["c1_buy"]);
        else if (x1 < 0) cost -= pkg["c1"] * (-x1) * (1 - comm["c1_sell"]);

        if (x2 > 0) cost += pkg["c2"] * x2 * (1 + comm["c2_buy"]);
        else if (x2 < 0) cost -= pkg["c2"] * (-x2) * (1 - comm["c2_sell"]);

        if (x3 > 0) cost += pkg["d"] * x3 * (1 + comm["d_buy"]);
        else if (x3 < 0) cost -= pkg["d"] * (-x3) * (1 - comm["d_sell"]);

        return cost;
    }


    static State Transition(int stage, State s, Action a, string key)
    {
        var m = stageData[stage][key];

        int nc1 = (int)Math.Round((s.c1 + pkg["c1"] * a.x1) * m.k1);
        int nc2 = (int)Math.Round((s.c2 + pkg["c2"] * a.x2) * m.k2);
        int nd  = (int)Math.Round((s.d  + pkg["d"]  * a.x3) * m.kd);
        int nf  = (int)Math.Round(s.f - ActionCost(a.x1, a.x2, a.x3));

        return new State(nc1, nc2, nd, nf);
    }

}