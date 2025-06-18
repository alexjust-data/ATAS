"""Kelly Simulator v4_1 – Interactive Risk Dashboard
────────────────────────────────────────────────────────────────────────────
• Tabs: 📋 Riesgo | 📈 Expectativa | 🔥 Rachas | 📉 Curvas | 🧮 Monte Carlo 
• Drawdown / Umbral de ruina configurable  
• Modo Determinista (analítico) o Estocástico (Monte Carlo)  
"""

# ─────────────────────────
# 0. Imports y config
# ─────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import beta
from ipywidgets import (
    FloatSlider, IntSlider, Checkbox, Dropdown, Output,
    VBox, HBox, Tab, interactive_output
)
import core.global_state as gs    # ← tu DataFrame de trades en gs.df

plt.rcParams["figure.dpi"] = 110
sns.set_style("whitegrid")

# ─────────────────────────
# 1. Helpers
# ─────────────────────────
def _norm(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        df = df.sort_values("date")
    return df.reset_index(drop=True)

# ─────────────────────────
# 2. Sliders & widgets
# ─────────────────────────
def session_stats():
    try:
        df = _norm(gs.df)
        win_p = (df["pnl_net"] > 0).mean()
        gain  = df.loc[df["pnl_net"] > 0, "pnl_net"].sum()
        loss  = df.loc[df["pnl_net"] < 0, "pnl_net"].sum()
        r     = gain / abs(loss) if loss else 1.0
        cap   = df["equity"].iloc[-1]
        dd    = (1 - df["equity"].div(df["equity"].cummax()).min()) * 100
        return win_p, r, cap, dd
    except Exception:
        return 0.55, 2.0, 1_000.0, 25.0

P0, R0, CAP0, DD0 = session_stats()

# Básicos
P_sl      = FloatSlider(value=P0,   min=0,   max=1,     step=0.001, description="Win %")
R_sl      = FloatSlider(value=R0,   min=0.1, max=10,    step=0.01,  description="Win/Loss R")
cap_sl    = FloatSlider(value=CAP0, min=100, max=20_000,step=1,     description="Capital $")
frac_sl   = FloatSlider(value=1.0,  min=0.1, max=2,     step=0.05,  description="% Kelly")
trades_sl = IntSlider( value=25,    min=10,  max=200,                description="# Trades")
racha_sl  = IntSlider( value=5,     min=2,   max=20,                description="Racha ≥")

# Umbral de ruina (% de capital inicial)
ruina_sl  = FloatSlider(value=0.5,  min=0.1, max=0.9,  step=0.01,
                        description="Umbral ruina %", readout_format=".0%")

# Monte Carlo
paths_sl  = IntSlider(value=500, min=100, max=5000, step=100, description="# Paths")
log_ck    = Checkbox(False, description="Log scale")
out_ck    = Checkbox(True,  description="Show outliers")

# Flags
bayes_ck  = Checkbox(True,  description="Ajuste Bayesiano")
markov_ck = Checkbox(True,  description="Penalización Markov")

# Modo de simulación
mode_dd   = Dropdown(options=[("Determinista", "det"),
                              ("Estocástico (MC)", "mc")],
                     value="det", description="Modo")

# ─────────────────────────
# 3. Probabilidades & Kelly
# ─────────────────────────
def _prob_no_streak(p_loss: float, n: int, L: int) -> float:
    state = np.zeros(L); state[0] = 1
    for _ in range(n):
        new = np.zeros_like(state)
        new[0] += (1 - p_loss) * state.sum()
        new[1:] += p_loss * state[:-1]
        state = new
    return state.sum()

def loss_streak_prob(p_loss: float, n: int, L: int) -> float:
    return 1 - _prob_no_streak(p_loss, n, L)

def win_streak_prob(p_win: float, n: int, L: int) -> float:
    return loss_streak_prob(1 - p_win, n, L)

def kelly_puro(p: float, r: float) -> float:
    return max(p - (1 - p) / r, 0)

def kelly_markov_adj(p: float, r: float, trans: pd.DataFrame) -> float:
    phi = trans.loc["loss","loss"] if "loss" in trans.index else 0.0
    return kelly_puro(p,r) * (1 - phi)

def kelly_penalizado(k: float, p_neg: float) -> float:
    return max(0, k / (1 + p_neg))

# ─────────────────────────
# 4. Streak Edge Δ
# ─────────────────────────
def streak_edge(trans: pd.DataFrame, n: int, L: int) -> float:
    if {"win","loss"}.issubset(trans.index):
        p_w = trans.loc["win","win"]
        p_l = trans.loc["loss","loss"]
        return win_streak_prob(p_w,n,L) - loss_streak_prob(p_l,n,L)
    return np.nan

# ─────────────────────────
# 5. Tabla de pérdidas condicional
# ─────────────────────────
def prob_win_cond(p_prior,w,l,cap_now,cap_init,bayes=True):
    if bayes:
        alpha,beta_ = 1+w,1+l
        ep = alpha/(alpha+beta_)
        lo,hi = beta.ppf([0.025,0.975],alpha,beta_)
    else:
        ep,lo,hi = p_prior,np.nan,np.nan
    ep += (cap_now/cap_init - 1)*0.1
    return max(0,min(ep,1)), lo, hi

def tabla_perdidas_dinamica(cap0,pct,p,r,L,bayes,trans,wins0,losses0,N=10):
    rows=[]; cap=cap0
    for i in range(1,N+1):
        risk=cap*pct; cap-=risk
        p_c,lo,hi=prob_win_cond(p,wins0,losses0+i,cap,cap0,bayes)
        st = loss_streak_prob(1-p,N-i+1,L); ed = streak_edge(trans,N-i+1,L)
        rows.append([i,round(cap,2),round(risk,2),
                     f"{p_c:.2%}",f"{1-p_c:.2%}",
                     f"{1-st:.2%}",f"{st:.2%}",
                     f"{lo:.2%}" if not np.isnan(lo) else "-",
                     f"{hi:.2%}" if not np.isnan(hi) else "-",
                     f"{ed:.2%}" if not np.isnan(ed) else "-"])
    return pd.DataFrame(rows,columns=[
        "#","Capital tras pérdida","Riesgo $","Prob. win cond.","Prob. loss cond.",
        "Prob. win racha","Prob. loss racha","IC 95 % inf","IC 95 % sup","Streak Edge (Δ)"
    ])

# ─────────────────────────
# 6. Gráficos deterministas
# ─────────────────────────
def graf_expectativa(cap0,pct,p_base,filas=10):
    caps,heur,bayes,lo,hi=[],[],[],[],[]
    cap=cap0; wins=losses=0
    for _ in range(filas):
        caps.append(cap)
        heur.append(prob_win_cond(p_base,0,0,cap,cap0,False)[0])
        p_b,l,h=prob_win_cond(p_base,wins,losses,cap,cap0,True)
        bayes.append(p_b); lo.append(l); hi.append(h)
        cap-=cap*pct; losses+=1
    plt.figure(figsize=(7,4))
    plt.plot(caps,heur,"--o",label="Heurística")
    plt.plot(caps,bayes,"-x",label="Bayes")
    plt.fill_between(caps,lo,hi,alpha=.25,label="IC 95 %")
    plt.ylim(0,1); plt.xlabel("Capital"); plt.ylabel("E[P]")
    plt.title("Probabilidad esperada de ganar vs capital")
    plt.legend(); plt.tight_layout(); plt.show()

def ruin_path(cap0,pct,max_dd):
    caps=[cap0]; pct/=100
    while caps[-1] > cap0*(1-max_dd) and len(caps)<1000:
        caps.append(caps[-1]*(1-pct))
    return caps

def graf_ruina(cap0,kelly_pct,max_dd):
    plt.figure(figsize=(8,4))
    for frac in [1,0.5,0.25]:
        pct=kelly_pct*frac*100
        if pct==0: continue
        path=ruin_path(cap0,pct,max_dd)
        n_tr=len(path)-1
        plt.plot(range(len(path)),path,label=f"{int(frac*100)} % Kelly ({n_tr} trades)")
        plt.axvline(n_tr,color=plt.gca().lines[-1].get_color(),ls=":",lw=1)
    plt.axhline(cap0*(1-max_dd),ls="--",color="#0077cc",
                label=f"Límite ruina ({int((1-max_dd)*100)}%)")
    plt.xlabel("Trades"); plt.ylabel("Capital")
    plt.title("Curvas de ruina (% Kelly)")
    plt.legend(); plt.tight_layout(); plt.show()

def _gen_streak_tbl(trades=50,max_L=11,kind="neg",p_step=5):
    rows=[]
    for w in range(p_step,100,p_step):
        row={"Win %":w}
        for L in range(2,max_L+1):
            pr = loss_streak_prob(1-w/100,trades,L) if kind=="neg" \
                 else win_streak_prob(w/100,trades,L)
            row[f">= {L}"]=pr
        rows.append(row)
    return pd.DataFrame(rows)

def heatmap_rachas(trades=50,max_L=11):
    df_w=_gen_streak_tbl(trades,max_L,"pos")
    df_l=_gen_streak_tbl(trades,max_L,"neg")
    df_w["Win %"] = df_w["Win %"].astype(int)
    df_l["Win %"] = df_l["Win %"].astype(int)
    fig,axs=plt.subplots(1,2,figsize=(16,6))
    sns.heatmap(df_w.set_index("Win %"),annot=True,fmt=".0%",cmap="RdYlGn",vmin=0,vmax=1,cbar=False,ax=axs[0])
    axs[0].set_title("Prob ≥L rachas GANADORAS"); axs[0].set_yticklabels(axs[0].get_yticklabels(),rotation=0)
    sns.heatmap(df_l.set_index("Win %"),annot=True,fmt=".0%",cmap="RdYlGn_r",vmin=0,vmax=1,cbar=False,ax=axs[1])
    axs[1].set_title("Prob ≥L rachas PERDEDORAS"); axs[1].set_yticklabels(axs[1].get_yticklabels(),rotation=0)
    plt.tight_layout(); plt.show()

# ─────────────────────────
# 7. Monte Carlo
# ─────────────────────────
def mc_paths(p,r,f,cap0,n,n_paths,seed=None):
    rng=np.random.default_rng(seed)
    eq=np.empty((n_paths,n+1)); eq[:,0]=cap0
    for t in range(n):
        wins = rng.random(n_paths) < p
        eq[:,t+1]=eq[:,t]+np.where(wins,eq[:,t]*f*r,-eq[:,t]*f)
    return eq

def mc_cvar5(eq):
    fin=eq[:,-1]; p5=np.percentile(fin,5)
    return float(fin[fin<=p5].mean()), float(p5)

def graf_mc(eq,log_scale=False,show_out=True):
    perc=[5,25,50,75,95]
    plt.figure(figsize=(10,5))
    for p in perc:
        plt.plot(np.percentile(eq,p,axis=0),label=f"{p}%")
    if show_out:
        idx=np.random.choice(eq.shape[0],min(20,eq.shape[0]),replace=False)
        plt.plot(eq[idx].T,alpha=.15,color="grey")
    if log_scale: plt.yscale("log")
    plt.title("Monte Carlo Simulation"); plt.xlabel("Trade #"); plt.ylabel("Capital")
    plt.legend(); plt.tight_layout(); plt.show()

# ─────────────────────────
# 8. Outputs
# ─────────────────────────
out_tbl,out_exp,out_streak,out_ruina,out_mc,out_sum=Output(),Output(),Output(),Output(),Output(),Output()

# ─────────────────────────
# 9. Función principal de actualización
# ─────────────────────────
def _update(P,R,cap0,frac,trades,L,max_dd,use_bayes,use_markov,n_paths,log_scale,show_out,modo):
    for o in [out_tbl,out_exp,out_streak,out_ruina,out_mc,out_sum]:
        with o: o.clear_output()

    if modo=="det":
        try:
            df=_norm(gs.df); df["result"]=np.where(df["pnl_net"]>0,"win","loss")
            trans=pd.crosstab(df["result"].shift(),df["result"],normalize=0).fillna(0)
            wins0=(df["pnl_net"]>0).sum(); losses0=len(df)-wins0
        except Exception:
            trans=pd.DataFrame({"win":{"win":P,"loss":1-P},"loss":{"win":P,"loss":1-P}})
            wins0=losses0=0

        k_p = kelly_puro(P,R)
        p_neg = loss_streak_prob(1-P,trades,L)
        k_adj = kelly_markov_adj(P,R,trans) if use_markov else kelly_penalizado(k_p,p_neg)
        pct_tr = k_adj*frac
        phi = trans.loc["loss","loss"] if "loss" in trans.index else 0.0

        with out_sum:
            print(f"- Prob. racha negativa: {p_neg*100:.2f}%")
            print(f"- Kelly % puro: {k_p*100:.4f}%")
            print(f"- Kelly ajust. Markov: {k_adj*100:.4f}% (φ={phi:.2%})")
            print(f"- Fracción seleccionada: {frac*100:.1f}%")
            print(f"⇒ Capital/trade ≈ ${cap0*pct_tr:.2f}")
            print("CVaR 5 %: Solo disponible en modo Monte Carlo")

        with out_tbl:
            display(tabla_perdidas_dinamica(cap0,pct_tr,P,R,L,use_bayes,trans,wins0,losses0))

        with out_exp: graf_expectativa(cap0,pct_tr,P)
        with out_streak: heatmap_rachas(trades,L)
        with out_ruina: graf_ruina(cap0,k_p,max_dd)

        with out_mc:
            eq=mc_paths(P,R,pct_tr,cap0,trades,n_paths)
            es,p5=mc_cvar5(eq)
            print(f"P = {P:.4f}   R = {R:.2f}   Kelly = {pct_tr:.2f}   Capital = {cap0}")
            print(f"CVaR 5 %: {es:.2f} (percentil 5 % = {p5:.2f})")
            graf_mc(eq,log_scale,show_out)

    else:
        pct_tr = frac
        eq = mc_paths(P,R,pct_tr,cap0,trades,n_paths)
        p_neg_emp = np.mean([loss_streak_prob(1-P,trades,L) for _ in range(1)])
        es,p5 = mc_cvar5(eq)

        with out_sum:
            print(f"- Prob. racha negativa (MC aprox): {p_neg_emp*100:.2f}%")
            print(f"- Fracción fija usada: {frac*100:.1f}%")
            print(f"⇒ Capital/trade ≈ ${cap0*frac:.2f}")
            print(f"CVaR 5 %: {es:.2f} (percentil 5 % = {p5:.2f})")

        with out_tbl: print("ℹ️ Tabla desactivada en modo MC")
        with out_exp: print("ℹ️ Expectativa no calculada en modo MC")
        with out_streak: heatmap_rachas(trades,L)
        with out_ruina: graf_ruina(cap0,frac,max_dd)
        with out_mc: graf_mc(eq,log_scale,show_out)

# ─────────────────────────
# 10. Interfaz interactiva
# ─────────────────────────
linked=interactive_output(_update,{
    "P":P_sl,"R":R_sl,"cap0":cap_sl,"frac":frac_sl,"trades":trades_sl,"L":racha_sl,
    "max_dd":ruina_sl,"use_bayes":bayes_ck,"use_markov":markov_ck,"n_paths":paths_sl,
    "log_scale":log_ck,"show_out":out_ck,"modo":mode_dd
})

# ─────────────────────────
# 11. Layout y pestañas
# ─────────────────────────
controls=VBox([
    HBox([mode_dd]),
    HBox([P_sl,R_sl,frac_sl]),
    HBox([cap_sl,trades_sl,racha_sl]),
    HBox([paths_sl,log_ck,out_ck]),
    HBox([bayes_ck,markov_ck]),
    HBox([ruina_sl]),
])

tabs=Tab(children=[VBox([out_sum,out_tbl]),out_exp,out_streak,out_ruina,out_mc])
tabs.set_title(0,"📋 Riesgo")
tabs.set_title(1,"📈 Expectativa")
tabs.set_title(2,"🔥 Rachas")
tabs.set_title(3,"📉 Curvas")
tabs.set_title(4,"🧮 Monte Carlo")

# ─────────────────────────
# 12. Mostrar interfaz
# ─────────────────────────
def mostrar_interfaz():
    display(VBox([controls,tabs,linked]))

if __name__=="__main__" or "get_ipython" in globals():
    mostrar_interfaz()











