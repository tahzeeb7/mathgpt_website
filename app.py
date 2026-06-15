"""
╔══════════════════════════════════════════════════════════════════╗
║         ADVANCED MATHGPT — by Muhammad Tahzeeb Shah              ║
║   Beats math-gpt.org with: SymPy CAS + Groq LLM + Wolfram API  ║
║   + LaTeX Rendering + Graphing + History + Multi-Mode Solving    ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import sympy as sp
from sympy import *
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
import numpy as np
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
from groq import Groq
import json
import re
import urllib.parse
import urllib.request
from datetime import datetime

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Advanced MathGPT — Tahzeeb Shah",
    page_icon="🧮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html,body,[class*="css"]{
    font-family:'Space Grotesk',sans-serif;
    background:#02060e;
    color:#e6edf3;
}

.main-header{
    background:linear-gradient(135deg,#061020,#0d1828,#061020);
    border:1px solid rgba(56,139,253,0.25);
    border-radius:16px;
    padding:28px 36px;
    margin-bottom:24px;
    position:relative;
    overflow:hidden;
}

.main-header::before{
    content:'';
    position:absolute;
    inset:0;
    background:radial-gradient(ellipse at 20% 50%,rgba(56,139,253,0.08) 0%,transparent 60%),
               radial-gradient(ellipse at 80% 50%,rgba(57,213,213,0.06) 0%,transparent 60%);
}

.main-header h1{
    font-size:2.4rem;
    font-weight:800;
    margin:0;
    background:linear-gradient(90deg,#58a6ff,#39d5d5,#bc8cff);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}

.main-header p{
    color:#8b949e;
    margin:6px 0 0;
    font-size:.95rem;
}

.result-card, .step-card{
    color:#e6edf3 !important;
}

/* Streamlit text fix */
.stMarkdown, .stMarkdown p, .stMarkdown div{
    color:#e6edf3 !important;
}

.stApp{
    color:#e6edf3 !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ─────────────────────────────────────────────────────────────────
# PARSER SETUP
# ─────────────────────────────────────────────────────────────────
TRANSFORMATIONS = (standard_transformations +
                   (implicit_multiplication_application, convert_xor,))

x, y, z, t, n, a, b, c = symbols('x y z t n a b c')

def safe_parse(expr_str: str):
    local_dict = {
        'pi': pi, 'e': E, 'E': E, 'I': I, 'oo': oo,
        'inf': oo, 'infinity': oo,
        **{s: Symbol(s) for s in list('xyztnabckmfgh')},
    }
    clean = (expr_str.replace('^','**').replace('×','*').replace('÷','/')
             .replace('√','sqrt').replace('∞','oo').replace('π','pi').strip())
    try:
        return parse_expr(clean, local_dict=local_dict, transformations=TRANSFORMATIONS)
    except Exception:
        return parse_expr(clean, local_dict=local_dict)

# ─────────────────────────────────────────────────────────────────
# SOLVER FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def solve_equation(expr_str, var_str="x"):
    steps = []
    var = Symbol(var_str.strip())
    if "=" in expr_str:
        lhs, rhs = expr_str.split("=", 1)
        lhs_expr = safe_parse(lhs)
        rhs_expr = safe_parse(rhs)
        expr = lhs_expr - rhs_expr
        steps.append(("Rearranged", f"${latex(lhs_expr)} - ({latex(rhs_expr)}) = 0$"))
    else:
        expr = safe_parse(expr_str)
        steps.append(("Expression", f"${latex(expr)} = 0$"))

    expanded = expand(expr)
    steps.append(("Expanded", f"${latex(expanded)} = 0$"))
    fac = factor(expr)
    if fac != expr:
        steps.append(("Factored", f"${latex(fac)} = 0$"))

    solutions = solve(expr, var)
    if not solutions:
        try:
            nsol = nsolve(expr, var, 0)
            solutions = [nsol]
            steps.append(("Numerical Root", f"${latex(nsol)}$"))
        except Exception:
            pass

    steps.append(("Solutions", f"${var} = {latex(solutions)}$"))
    verified = []
    for sol in solutions:
        check = simplify(expr.subs(var, sol))
        verified.append((sol, check == 0 or abs(complex(check)) < 1e-9))

    return {"type":"equation","expression":expr,"solutions":solutions,
            "verified":verified,"steps":steps,"latex":latex(solutions)}


def differentiate(expr_str, var_str="x", order=1):
    steps = []
    var = Symbol(var_str.strip())
    expr = safe_parse(expr_str)
    steps.append(("Function", f"$f({var}) = {latex(expr)}$"))
    result = expr
    for i in range(order):
        result = diff(result, var)
        steps.append((f"Derivative step {i+1}", f"${latex(result)}$"))
    simplified = simplify(result)
    if simplified != result:
        steps.append(("Simplified", f"${latex(simplified)}$"))
    critical_pts = solve(diff(expr, var), var)
    if critical_pts:
        steps.append(("Critical Points", f"${var} = {latex(critical_pts)}$"))
    return {"type":"derivative","expression":expr,"result":simplified,
            "steps":steps,"latex":latex(simplified),"critical_points":critical_pts}


def integrate_expr(expr_str, var_str="x", lower=None, upper=None):
    steps = []
    var = Symbol(var_str.strip())
    expr = safe_parse(expr_str)
    steps.append(("Integrand", f"$f({var}) = {latex(expr)}$"))
    if lower is not None and upper is not None:
        lo = safe_parse(str(lower))
        hi = safe_parse(str(upper))
        steps.append(("Bounds", f"$a={latex(lo)},\\ b={latex(hi)}$"))
        antideriv = integrate(expr, var)
        steps.append(("Antiderivative", f"$F({var}) = {latex(antideriv)} + C$"))
        result = simplify(integrate(expr, (var, lo, hi)))
        steps.append(("Evaluated", f"$\\int_{{{latex(lo)}}}^{{{latex(hi)}}} f d{var} = {latex(result)}$"))
        numeric_val = None
        try:
            numeric_val = float(result.evalf())
        except Exception:
            pass
        return {"type":"definite_integral","expression":expr,"result":result,
                "numeric":numeric_val,"steps":steps,"latex":latex(result)}
    else:
        antideriv = integrate(expr, var)
        steps.append(("Antiderivative", f"${latex(antideriv)} + C$"))
        check = simplify(diff(antideriv, var) - expr)
        steps.append(("Verification", f"$d/d{var}[F] - f = {latex(check)}$ {'✓' if check==0 else '≈0'}"))
        return {"type":"indefinite_integral","expression":expr,"result":antideriv,
                "steps":steps,"latex":latex(antideriv)+" + C"}


def compute_limit(expr_str, var_str="x", point_str="oo", direction="+-"):
    steps = []
    var = Symbol(var_str.strip())
    expr = safe_parse(expr_str)
    point = safe_parse(point_str)
    steps.append(("Expression", f"$\\lim_{{{var}\\to{latex(point)}}} {latex(expr)}$"))
    if direction == "+":
        result = limit(expr, var, point, "+")
        steps.append(("Right-hand limit", f"${latex(result)}$"))
    elif direction == "-":
        result = limit(expr, var, point, "-")
        steps.append(("Left-hand limit", f"${latex(result)}$"))
    else:
        rl = limit(expr, var, point, "+")
        ll = limit(expr, var, point, "-")
        result = limit(expr, var, point)
        steps.append(("Right limit", f"${latex(rl)}$"))
        steps.append(("Left limit", f"${latex(ll)}$"))
        if rl == ll:
            steps.append(("Limit Exists", f"${latex(result)}$"))
        else:
            steps.append(("DNE", "Left ≠ Right"))
    return {"type":"limit","expression":expr,"result":result,
            "steps":steps,"latex":latex(result)}


def matrix_ops(matrix_str):
    steps = []
    try:
        rows = json.loads(matrix_str)
        M = Matrix(rows)
    except Exception:
        M = eval(f"Matrix({matrix_str})", {"Matrix": Matrix,
                  **{s: Symbol(s) for s in 'abcdefghijklmnopqrstuvwxyz'}})
    steps.append(("Matrix", f"$A = {latex(M)}$"))
    steps.append(("Size", f"{M.rows} × {M.cols}"))
    results = {"matrix": M, "steps": steps, "type": "matrix"}
    if M.is_square:
        det_val = M.det()
        steps.append(("Determinant", f"$\\det(A) = {latex(det_val)}$"))
        results["det"] = det_val
        if det_val != 0:
            inv_M = M.inv()
            steps.append(("Inverse", f"$A^{{-1}} = {latex(inv_M)}$"))
            results["inverse"] = inv_M
        try:
            evals = M.eigenvals()
            steps.append(("Eigenvalues", f"${latex(evals)}$"))
            results["eigenvals"] = evals
        except Exception:
            pass
        tr = M.trace()
        steps.append(("Trace", f"$\\text{{tr}}(A) = {latex(tr)}$"))
        results["trace"] = tr
    results["rank"] = M.rank()
    steps.append(("Rank", str(results["rank"])))
    return results


def taylor_series(expr_str, var_str="x", point_str="0", order=6):
    var = Symbol(var_str.strip())
    expr = safe_parse(expr_str)
    point = safe_parse(point_str)
    series_expr = series(expr, var, point, order)
    return {
        "type": "taylor", "expression": expr, "result": series_expr,
        "steps": [
            ("Function", f"$f({var}) = {latex(expr)}$"),
            ("Expanded at", f"${var} = {latex(point)}$, order {order}"),
            ("Series", f"${latex(series_expr)}$"),
        ],
        "latex": latex(series_expr),
    }


def number_theory_analysis(n_str):
    steps = []
    n_val = int(float(n_str))
    steps.append(("Input", str(n_val)))
    factors = factorint(n_val)
    steps.append(("Prime Factorization",
                   " × ".join([f"{p}^{e}" if e>1 else str(p) for p,e in factors.items()])))
    is_prime_val = isprime(n_val)
    steps.append(("Is Prime?", "Yes ✓" if is_prime_val else "No"))
    divs = divisors(n_val)
    steps.append(("Divisors", str(divs)))
    phi = totient(n_val)
    steps.append(("Euler φ(n)", str(phi)))
    return {"type":"number_theory","n":n_val,"factors":factors,
            "is_prime":is_prime_val,"divisors":divs,"totient":phi,"steps":steps}


def simplify_expr(expr_str):
    steps = []
    expr = safe_parse(expr_str)
    steps.append(("Input", f"${latex(expr)}$"))
    simp = simplify(expr)
    steps.append(("Simplified", f"${latex(simp)}$"))
    exp_expr = expand(expr)
    if exp_expr != simp:
        steps.append(("Expanded", f"${latex(exp_expr)}$"))
    fac = factor(expr)
    if fac != simp:
        steps.append(("Factored", f"${latex(fac)}$"))
    trig = trigsimp(expr)
    if trig != simp:
        steps.append(("Trig Simplified", f"${latex(trig)}$"))
    return {"type":"simplify","expression":expr,"simplified":simp,
            "expanded":exp_expr,"factored":fac,"steps":steps,"latex":latex(simp)}

# ─────────────────────────────────────────────────────────────────
# GRAPHING
# ─────────────────────────────────────────────────────────────────

def plot_function(expr_str, x_min=-10, x_max=10, title=None, extra_exprs=None):
    var = Symbol('x')
    exprs = [safe_parse(expr_str)]
    labels = [f"f(x) = {expr_str}"]
    if extra_exprs:
        for e in extra_exprs:
            exprs.append(safe_parse(e))
            labels.append(f"g(x) = {e}")
    x_vals = np.linspace(x_min, x_max, 800)
    colors = ['#58a6ff','#39d5d5','#bc8cff','#56d364','#f0c060','#ff7b72']
    fig = go.Figure()
    for idx, (expr, label) in enumerate(zip(exprs, labels)):
        try:
            f_lam = lambdify(var, expr, modules=['numpy'])
            y_vals = f_lam(x_vals)
            y_vals = np.where(np.abs(np.real(y_vals.astype(complex))) > 1e6, np.nan, np.real(y_vals.astype(complex)))
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', name=label,
                                      line=dict(color=colors[idx % len(colors)], width=2.5)))
        except Exception as err:
            st.warning(f"Could not plot {label}: {err}")
    fig.update_layout(
        title=title or f"Graph: {expr_str}",
        paper_bgcolor='#02060e', plot_bgcolor='#080f1c',
        font=dict(color='#e6edf3', family='Space Grotesk'),
        xaxis=dict(gridcolor='rgba(56,139,253,0.12)', zerolinecolor='rgba(56,139,253,0.4)', color='#8b949e'),
        yaxis=dict(gridcolor='rgba(56,139,253,0.12)', zerolinecolor='rgba(56,139,253,0.4)', color='#8b949e'),
        legend=dict(bgcolor='rgba(8,15,28,0.8)', bordercolor='rgba(56,139,253,0.3)', borderwidth=1),
        hovermode='x unified',
    )
    return fig


def plot_3d(expr_str, x_range=(-5,5), y_range=(-5,5)):
    xv, yv = symbols('x y')
    expr = safe_parse(expr_str)
    xa = np.linspace(*x_range, 80)
    ya = np.linspace(*y_range, 80)
    X, Y = np.meshgrid(xa, ya)
    f = lambdify((xv, yv), expr, modules=['numpy'])
    try:
        Z = f(X, Y)
        Z = np.where(np.abs(np.real(Z.astype(complex))) > 1e6, np.nan, np.real(Z.astype(complex)))
    except Exception as err:
        st.error(f"3D plot error: {err}")
        return None
    fig = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='Blues', opacity=0.9)])
    fig.update_layout(
        title=f"z = {expr_str}", paper_bgcolor='#02060e', font=dict(color='#e6edf3'),
        scene=dict(
            xaxis=dict(backgroundcolor='#080f1c', gridcolor='rgba(56,139,253,0.2)', color='#8b949e'),
            yaxis=dict(backgroundcolor='#080f1c', gridcolor='rgba(56,139,253,0.2)', color='#8b949e'),
            zaxis=dict(backgroundcolor='#080f1c', gridcolor='rgba(56,139,253,0.2)', color='#8b949e'),
        )
    )
    return fig

# ─────────────────────────────────────────────────────────────────
# GROQ LLM
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an Advanced Mathematics Professor AI created by Muhammad Tahzeeb Shah (BS Mathematics, Pakistan).

You are a deep expert in:
• Pure Mathematics: Real Analysis, Abstract Algebra, Topology, Number Theory, Complex Analysis
• Applied Mathematics: Statistics, Probability, Optimisation, Numerical Methods, Fuzzy Logic
• CS Mathematics: Algorithms, Graph Theory, Cryptography, Game Theory
• ML Mathematics: Linear Algebra, Calculus, Information Theory

Rules for every answer:
1. Step-by-step rigorous solution
2. Explain MATHEMATICAL INTUITION at each step
3. Mention relevant theorems and lemmas
4. State domain restrictions and edge cases
5. Suggest alternative methods when they exist
6. End with a "Mathematical Insight" section

Structure every response:
- **Problem Setup**
- **Step-by-step Solution**
- **Key Concepts Used**
- **Mathematical Insight**"""


def get_groq_explanation(question, context="", model="llama-3.3-70b-versatile"):
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        client = Groq(api_key=api_key)
    except Exception:
        return "⚠️ Add `GROQ_API_KEY` to `.streamlit/secrets.toml` to enable AI explanations."
    messages = [{"role":"system","content":SYSTEM_PROMPT}]
    content = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    messages.append({"role":"user","content":content})
    try:
        resp = client.chat.completions.create(model=model, messages=messages,
                                               temperature=0.2, max_tokens=2048)
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq error: {e}"


def wolfram_query(query):
    try:
        app_id = st.secrets.get("WOLFRAM_APP_ID", "")
        if not app_id:
            return {"error": "Set WOLFRAM_APP_ID in secrets.toml"}
        encoded = urllib.parse.quote_plus(query)
        url = (f"http://api.wolframalpha.com/v2/query?"
               f"appid={app_id}&input={encoded}&format=plaintext&output=JSON")
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read().decode())
        pods = data.get("queryresult",{}).get("pods",[])
        result = {}
        for pod in pods:
            title = pod.get("title","")
            texts = [sp.get("plaintext","") for sp in pod.get("subpods",[]) if sp.get("plaintext")]
            if texts:
                result[title] = "\n".join(texts)
        return result
    except Exception as e:
        return {"error": str(e)}

# ─────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────

def show_steps(steps):
    for i, (label, content) in enumerate(steps):
        st.markdown(f"""
        <div class="step-card">
            <span class="badge badge-cyan">Step {i+1}</span>
            <strong style="color:#e6edf3;margin-left:8px">{label}</strong><br>
            <div style="margin-top:6px;color:#8b949e;font-size:.9rem">{content}</div>
        </div>""", unsafe_allow_html=True)


def add_history(query, mode, result_latex=""):
    st.session_state.history.insert(0, {
        "time": datetime.now().strftime("%H:%M"),
        "query": query, "mode": mode, "latex": result_latex,
    })
    st.session_state.history = st.session_state.history[:20]

# ═══════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:12px 0 20px">
        <div style="font-size:2.2rem">🧮</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:.85rem;color:#58a6ff;font-weight:700">ADVANCED MATHGPT</div>
        <div style="font-size:.72rem;color:#484f58;margin-top:2px">by Muhammad Tahzeeb Shah</div>
    </div>""", unsafe_allow_html=True)

    mode = st.selectbox("Mode", [
        "🤖 AI Chat (Groq LLM)",
        "📐 Equation Solver",
        "∫ Integration",
        "∂ Differentiation",
        "lim Limits",
        "📊 Matrix Operations",
        "📈 Function Grapher",
        "🌐 3D Surface Grapher",
        "🔢 Taylor Series",
        "🔐 Number Theory",
        "✏️ Simplify / Factor",
        "🔭 Wolfram Alpha",
    ])

    st.markdown('<div class="math-divider"></div>', unsafe_allow_html=True)
    llm_model = st.selectbox("Groq LLM Model", [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "mixtral-8x7b-32768",
        "gemma2-9b-it",
    ])

    st.markdown('<div class="math-divider"></div>', unsafe_allow_html=True)
    with st.expander("📖 Input Syntax"):
        st.markdown("""
**Powers:** `x**2` (not x^2)
**Multiply:** `3*x` (not 3x)
**Functions:** `sin(x)`, `cos(x)`, `exp(x)`, `log(x)`, `sqrt(x)`
**Constants:** `pi`, `E`, `oo` (infinity), `I` (imaginary)

**Examples:**
- `x**3 - 6*x**2 + 11*x - 6 = 0`
- `exp(-x**2) * sin(x)`
- `[[1,2],[3,4]]` (matrix)
- `(x**2-1)/(x-1)` (simplify)
        """)
    with st.expander("🔑 API Keys Setup"):
        st.markdown("""
Create `.streamlit/secrets.toml`:
```toml
GROQ_API_KEY = "gsk_..."
WOLFRAM_APP_ID = "XXXX-YYYY"
```
- **Groq (free):** console.groq.com
- **Wolfram (free):** developer.wolframalpha.com
        """)

    st.markdown('<div class="math-divider"></div>', unsafe_allow_html=True)
    if st.session_state.history:
        st.markdown("### 📜 History")
        for item in st.session_state.history[:6]:
            st.markdown(f"""
            <div class="hist-item">
                <span style="color:#484f58">{item['time']}</span>
                <span class="badge badge-blue" style="margin-left:6px">{item['mode']}</span><br>
                {item['query'][:44]}{'…' if len(item['query'])>44 else ''}
            </div>""", unsafe_allow_html=True)
        if st.button("🗑️ Clear"):
            st.session_state.history = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<div class="main-header">
    <h1>🧮 Advanced MathGPT</h1>
    <p>SymPy CAS · Groq LLM · Wolfram Alpha · Plotly Graphs · LaTeX Rendering<br>
    <span style="color:#56d364;font-size:.82rem">✓ Equations &nbsp;✓ Calculus &nbsp;✓ Matrices &nbsp;✓ Number Theory &nbsp;✓ Taylor Series &nbsp;✓ 3D Graphs &nbsp;✓ AI Explanations</span></p>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# MODE ROUTING
# ═══════════════════════════════════════════════════════════════════

# ── AI CHAT ──────────────────────────────────────────────────────
if mode == "🤖 AI Chat (Groq LLM)":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask Any Math Question")
    question = st.text_area("Question", height=110, label_visibility="collapsed",
        placeholder="e.g. Prove √2 is irrational. Explain Fundamental Theorem of Calculus. What is the Riemann Hypothesis?")
    ask_btn = st.button("🚀 Ask MathGPT", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if ask_btn and question.strip():
        with st.spinner("🧠 Thinking deeply..."):
            answer = get_groq_explanation(question, model=llm_model)
        st.markdown(f'<div class="result-card">{answer}</div>', unsafe_allow_html=True)
        add_history(question, "AI Chat")

    st.markdown("#### 💡 Example Questions")
    examples = [
        "Prove the Fundamental Theorem of Calculus",
        "Explain RSA cryptosystem using number theory",
        "Prove there are infinitely many primes (Euclid's proof)",
        "What is Nash Equilibrium? Give a game theory example",
        "Explain eigenvectors geometrically",
        "Derive the quadratic formula from scratch",
        "What is Bayes' theorem? Medical testing example",
        "Explain P vs NP problem intuitively",
    ]
    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i%2].button(f"📌 {ex}", key=f"ex_{i}"):
            with st.spinner("🧠 Thinking..."):
                ans = get_groq_explanation(ex, model=llm_model)
            st.markdown(f'<div class="result-card">{ans}</div>', unsafe_allow_html=True)
            add_history(ex, "AI Chat")

# ── EQUATION SOLVER ──────────────────────────────────────────────
elif mode == "📐 Equation Solver":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 📐 Equation Solver  ·  Supports polynomial, trigonometric, exponential, systems")
    col1, col2 = st.columns([3,1])
    eq_input = col1.text_input("Equation", placeholder="x**2 - 5*x + 6 = 0", label_visibility="collapsed")
    var_input = col2.text_input("Variable", value="x", label_visibility="collapsed")
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    solve_btn = st.button("🔍 Solve", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if solve_btn and eq_input.strip():
        try:
            with st.spinner("⚙️ Computing..."):
                res = solve_equation(eq_input, var_input)
            tabs = st.tabs(["✅ Solution", "📋 Steps", "📊 Graph", "🤖 AI Explanation"])
            with tabs[0]:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f'<span class="badge badge-green">Solutions</span>', unsafe_allow_html=True)
                for sol, verified in res["verified"]:
                    icon = "✅" if verified else "⚠️"
                    st.latex(f"{var_input} = {latex(sol)}")
                    st.markdown(f"<small style='color:#8b949e'>{icon} Verified by substitution</small>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with tabs[1]:
                show_steps(res["steps"])
            with tabs[2]:
                try:
                    expr_for_plot = eq_input.split("=")[0].strip()
                    fig = plot_function(expr_for_plot, -10, 10, f"Graph: {eq_input}")
                    for sol in res["solutions"]:
                        try:
                            sf = float(sol.evalf())
                            fig.add_vline(x=sf, line=dict(color='#56d364', dash='dash', width=1.5),
                                          annotation_text=f"x={sf:.3f}")
                        except Exception:
                            pass
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Graph unavailable: {e}")
            with tabs[3]:
                if ai_explain:
                    with st.spinner("🤖 Explaining..."):
                        expl = get_groq_explanation(
                            f"Explain step-by-step how to solve: {eq_input}",
                            context=f"Solutions: {res.get('solutions','')}",
                            model=llm_model)
                    st.markdown(expl)
            add_history(eq_input, "Equation", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")
            st.info("Use `**` for powers, `*` for multiply. Example: `x**2 - 5*x + 6 = 0`")

# ── INTEGRATION ──────────────────────────────────────────────────
elif mode == "∫ Integration":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### ∫ Integration  ·  Definite & Indefinite")
    col1, col2 = st.columns([3,1])
    int_input = col1.text_input("f(x)", placeholder="x**2 * sin(x)", label_visibility="collapsed")
    int_var = col2.text_input("Variable", value="x", label_visibility="collapsed")
    int_type = st.radio("Type", ["Indefinite ∫ f dx", "Definite ∫ₐᵇ f dx"], horizontal=True)
    lower_b = upper_b = None
    if int_type == "Definite ∫ₐᵇ f dx":
        c1, c2 = st.columns(2)
        lower_b = c1.text_input("Lower a", value="0")
        upper_b = c2.text_input("Upper b", value="1")
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    int_btn = st.button("∫ Integrate", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if int_btn and int_input.strip():
        try:
            with st.spinner("⚙️ Integrating..."):
                res = integrate_expr(int_input, int_var, lower_b, upper_b)
            tabs = st.tabs(["✅ Result", "📋 Steps", "📊 Graph", "🤖 AI"])
            with tabs[0]:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<span class="badge badge-purple">Result</span>', unsafe_allow_html=True)
                st.latex(res["latex"])
                if res.get("numeric") is not None:
                    st.markdown(f"**Numeric:** `{res['numeric']:.10f}`")
                st.markdown('</div>', unsafe_allow_html=True)
            with tabs[1]:
                show_steps(res["steps"])
            with tabs[2]:
                try:
                    fig = plot_function(int_input, -5, 5, f"Integrand: {int_input}")
                    if lower_b and upper_b:
                        try:
                            lo, hi = float(lower_b), float(upper_b)
                            fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(88,166,255,0.1)",
                                          line_width=0, annotation_text="Integration Region")
                        except Exception:
                            pass
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Graph error: {e}")
            with tabs[3]:
                if ai_explain:
                    with st.spinner("🤖 Explaining..."):
                        expl = get_groq_explanation(
                            f"Explain the integration of {int_input} step by step",
                            context=f"Result: {res['latex']}", model=llm_model)
                    st.markdown(expl)
            add_history(f"∫{int_input}d{int_var}", "Integration", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")

# ── DIFFERENTIATION ──────────────────────────────────────────────
elif mode == "∂ Differentiation":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### ∂ Differentiation  ·  nth Order Derivatives")
    col1, col2, col3 = st.columns([3,1,1])
    diff_input = col1.text_input("f(x)", placeholder="sin(x) * exp(-x**2)", label_visibility="collapsed")
    diff_var = col2.text_input("Variable", value="x", label_visibility="collapsed")
    diff_order = col3.number_input("Order", min_value=1, max_value=10, value=1)
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    diff_btn = st.button("∂ Differentiate", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if diff_btn and diff_input.strip():
        try:
            with st.spinner("⚙️ Differentiating..."):
                res = differentiate(diff_input, diff_var, diff_order)
            tabs = st.tabs(["✅ Result", "📋 Steps", "📊 Graph", "🤖 AI"])
            with tabs[0]:
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown('<span class="badge badge-cyan">Derivative</span>', unsafe_allow_html=True)
                st.latex(f"\\frac{{d^{{{diff_order}}}}}{{d{diff_var}^{{{diff_order}}}}}\\left[{latex(res['expression'])}\\right] = {res['latex']}")
                if res.get("critical_points"):
                    st.markdown(f"**Critical points:** ${diff_var} = {latex(res['critical_points'])}$")
                st.markdown('</div>', unsafe_allow_html=True)
            with tabs[1]:
                show_steps(res["steps"])
            with tabs[2]:
                try:
                    fig = plot_function(diff_input, -5, 5, "f(x) and f'(x)",
                                        extra_exprs=[str(res["result"])])
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Graph error: {e}")
            with tabs[3]:
                if ai_explain:
                    with st.spinner("🤖 Explaining..."):
                        expl = get_groq_explanation(
                            f"Differentiate {diff_input} with respect to {diff_var}, order {diff_order}",
                            context=f"Result: {res['latex']}", model=llm_model)
                    st.markdown(expl)
            add_history(f"d/d{diff_var}[{diff_input}]", "Derivative", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")

# ── LIMITS ───────────────────────────────────────────────────────
elif mode == "lim Limits":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### lim Limits  ·  One-sided & Two-sided")
    col1, col2, col3, col4 = st.columns([3,1,1,1])
    lim_input = col1.text_input("f(x)", placeholder="sin(x)/x", label_visibility="collapsed")
    lim_var = col2.text_input("Variable", value="x", label_visibility="collapsed")
    lim_point = col3.text_input("→", value="0", label_visibility="collapsed")
    lim_dir = col4.selectbox("Dir", ["±", "+", "-"], label_visibility="collapsed")
    dir_map = {"±":"+-", "+":"+", "-":"-"}
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    lim_btn = st.button("lim Compute", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if lim_btn and lim_input.strip():
        try:
            with st.spinner("⚙️ Computing..."):
                res = compute_limit(lim_input, lim_var, lim_point, dir_map[lim_dir])
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge badge-gold">Limit Result</span>', unsafe_allow_html=True)
            st.latex(f"\\lim_{{{lim_var}\\to{lim_point}}} {latex(res['expression'])} = {res['latex']}")
            st.markdown('</div>', unsafe_allow_html=True)
            show_steps(res["steps"])
            try:
                fig = plot_function(lim_input, -5, 5, f"lim as {lim_var}→{lim_point}")
                try:
                    pt = float(safe_parse(lim_point).evalf())
                    fig.add_vline(x=pt, line=dict(color='#f0c060', dash='dash', width=1.5),
                                  annotation_text=f"{lim_var}→{lim_point}")
                except Exception:
                    pass
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
            if ai_explain:
                with st.spinner("🤖 Explaining..."):
                    expl = get_groq_explanation(
                        f"Evaluate lim({lim_var}→{lim_point}) {lim_input}",
                        context=f"Result: {res['latex']}", model=llm_model)
                st.markdown(expl)
            add_history(f"lim({lim_var}→{lim_point}) {lim_input}", "Limit", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")

# ── MATRIX OPERATIONS ────────────────────────────────────────────
elif mode == "📊 Matrix Operations":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 📊 Matrix Operations  ·  Det · Inv · Eigenvalues · Rank")
    st.markdown("Enter as JSON: `[[1,2],[3,4]]`")
    mat_input = st.text_area("Matrix", placeholder='[[2,1,-1],[1,3,2],[0,1,4]]',
                               height=80, label_visibility="collapsed")
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    mat_btn = st.button("📊 Compute", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if mat_btn and mat_input.strip():
        try:
            with st.spinner("⚙️ Computing..."):
                res = matrix_ops(mat_input)
            tabs = st.tabs(["📊 Results", "📋 Steps", "🤖 AI"])
            with tabs[0]:
                M = res["matrix"]
                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.latex(f"A = {latex(M)}")
                st.markdown('</div>', unsafe_allow_html=True)
                col1, col2 = st.columns(2)
                with col1:
                    if "det" in res:
                        st.markdown(f"**Determinant:** $\\det(A) = {latex(res['det'])}$")
                    if "trace" in res:
                        st.markdown(f"**Trace:** $\\text{{tr}}(A) = {latex(res['trace'])}$")
                    st.markdown(f"**Rank:** {res['rank']}")
                with col2:
                    if "eigenvals" in res:
                        st.markdown("**Eigenvalues:**")
                        for ev, mult in res["eigenvals"].items():
                            st.markdown(f"$\\lambda = {latex(ev)}$ (mult. {mult})")
                if "inverse" in res:
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("**Inverse:**")
                    st.latex(f"A^{{-1}} = {latex(res['inverse'])}")
                    st.markdown('</div>', unsafe_allow_html=True)
            with tabs[1]:
                show_steps(res["steps"])
            with tabs[2]:
                if ai_explain:
                    with st.spinner("🤖 Explaining..."):
                        expl = get_groq_explanation(
                            f"Explain all properties of this matrix: {mat_input}",
                            model=llm_model)
                    st.markdown(expl)
            add_history(mat_input, "Matrix", f"det={res.get('det','?')}")
        except Exception as e:
            st.error(f"❌ {e}")
            st.info("Example: `[[1,2,3],[4,5,6],[7,8,9]]`")

# ── FUNCTION GRAPHER ─────────────────────────────────────────────
elif mode == "📈 Function Grapher":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 📈 Function Grapher  ·  Plot multiple functions")
    f1 = st.text_input("f(x)", placeholder="sin(x) * exp(-x/5)", label_visibility="collapsed")
    extra_raw = st.text_input("Extra functions (comma-separated)",
                               placeholder="x**2/10, cos(x)", label_visibility="collapsed")
    col1, col2 = st.columns(2)
    x_min = col1.number_input("x min", value=-10.0)
    x_max = col2.number_input("x max", value=10.0)
    plot_btn = st.button("📈 Plot", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if plot_btn and f1.strip():
        try:
            extras = [e.strip() for e in extra_raw.split(",") if e.strip()] if extra_raw else None
            with st.spinner("📊 Plotting..."):
                fig = plot_function(f1, x_min, x_max, extra_exprs=extras)
            st.plotly_chart(fig, use_container_width=True)
            var = Symbol('x')
            expr = safe_parse(f1)
            col1, col2, col3 = st.columns(3)
            try:
                crits = solve(diff(expr, var), var)
                col1.markdown(f"**Critical pts:** {[f'{float(c.evalf()):.3f}' for c in crits if c.is_real]}")
            except Exception:
                pass
            try:
                roots = solve(expr, var)
                col2.markdown(f"**Roots:** {[f'{float(r.evalf()):.3f}' for r in roots if r.is_real]}")
            except Exception:
                pass
            try:
                col3.markdown(f"**y-int:** `{float(expr.subs(var, 0).evalf()):.4f}`")
            except Exception:
                pass
            add_history(f1, "Graph")
        except Exception as e:
            st.error(f"❌ {e}")

# ── 3D GRAPHER ───────────────────────────────────────────────────
elif mode == "🌐 3D Surface Grapher":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🌐 3D Surface Grapher  ·  f(x,y)")
    f3d = st.text_input("f(x,y)", placeholder="sin(x)*cos(y)",
                          label_visibility="collapsed")
    col1, col2, col3, col4 = st.columns(4)
    xmin3 = col1.number_input("x min", value=-5.0)
    xmax3 = col2.number_input("x max", value=5.0)
    ymin3 = col3.number_input("y min", value=-5.0)
    ymax3 = col4.number_input("y max", value=5.0)
    plot3_btn = st.button("🌐 Plot 3D", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if plot3_btn and f3d.strip():
        with st.spinner("🌐 Rendering 3D surface..."):
            fig = plot_3d(f3d, (xmin3, xmax3), (ymin3, ymax3))
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            add_history(f"3D: {f3d}", "3D Graph")

# ── TAYLOR SERIES ────────────────────────────────────────────────
elif mode == "🔢 Taylor Series":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🔢 Taylor / Maclaurin Series")
    col1, col2, col3, col4 = st.columns([3,1,1,1])
    ts_input = col1.text_input("f(x)", placeholder="sin(x)", label_visibility="collapsed")
    ts_var = col2.text_input("Variable", value="x", label_visibility="collapsed")
    ts_point = col3.text_input("About x₀", value="0", label_visibility="collapsed")
    ts_order = col4.number_input("Order", min_value=2, max_value=20, value=6)
    ts_btn = st.button("🔢 Expand", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if ts_btn and ts_input.strip():
        try:
            with st.spinner("⚙️ Expanding..."):
                res = taylor_series(ts_input, ts_var, ts_point, ts_order)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge badge-purple">Taylor Series</span>', unsafe_allow_html=True)
            st.latex(res["latex"])
            st.markdown('</div>', unsafe_allow_html=True)
            show_steps(res["steps"])
            try:
                var = Symbol(ts_var)
                approx_expr = res["result"].removeO()
                fig = plot_function(ts_input, -5, 5,
                    f"f(x) vs Taylor approx (order {ts_order})",
                    extra_exprs=[str(approx_expr)])
                st.plotly_chart(fig, use_container_width=True)
            except Exception:
                pass
            add_history(f"Taylor {ts_input}@{ts_point}", "Taylor", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")

# ── NUMBER THEORY ────────────────────────────────────────────────
elif mode == "🔐 Number Theory":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🔐 Number Theory  ·  Primes · Factorization · Euler φ · Divisors")
    nt_input = st.text_input("Integer", placeholder="360", label_visibility="collapsed")
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    nt_btn = st.button("🔐 Analyze", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if nt_btn and nt_input.strip():
        try:
            with st.spinner("⚙️ Analyzing..."):
                res = number_theory_analysis(nt_input)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<span class="badge badge-gold">Number Theory</span>', unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("Is Prime", "Yes ✓" if res["is_prime"] else "No")
            col2.metric("Euler φ(n)", str(res["totient"]))
            col3.metric("# Divisors", str(len(res["divisors"])))
            fac_str = " × ".join([f"{p}^{e}" if e>1 else str(p) for p,e in res["factors"].items()])
            st.markdown(f"**Factorization:** `{res['n']} = {fac_str}`")
            st.markdown(f"**Divisors:** `{res['divisors']}`")
            st.markdown('</div>', unsafe_allow_html=True)
            show_steps(res["steps"])
            if ai_explain:
                with st.spinner("🤖 Explaining..."):
                    expl = get_groq_explanation(
                        f"Explain number theory properties of {nt_input}",
                        context=f"Factors:{res['factors']}, φ={res['totient']}, prime={res['is_prime']}",
                        model=llm_model)
                st.markdown(expl)
            add_history(nt_input, "Number Theory", str(res["factors"]))
        except Exception as e:
            st.error(f"❌ {e}")

# ── SIMPLIFY / FACTOR ────────────────────────────────────────────
elif mode == "✏️ Simplify / Factor":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### ✏️ Simplify / Factor / Expand")
    simp_input = st.text_input("Expression",
        placeholder="(x**2-1)/(x-1)  or  sin(x)**2 + cos(x)**2",
        label_visibility="collapsed")
    ai_explain = st.checkbox("🤖 AI Explanation", value=True)
    simp_btn = st.button("✏️ Process", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if simp_btn and simp_input.strip():
        try:
            with st.spinner("⚙️ Processing..."):
                res = simplify_expr(simp_input)
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            col1.markdown("**Simplified:**"); col1.latex(res["latex"])
            col2.markdown("**Factored:**"); col2.latex(latex(res["factored"]))
            st.markdown("**Expanded:**"); st.latex(latex(res["expanded"]))
            st.markdown('</div>', unsafe_allow_html=True)
            show_steps(res["steps"])
            if ai_explain:
                with st.spinner("🤖 Explaining..."):
                    expl = get_groq_explanation(
                        f"Explain the simplification of {simp_input}",
                        context=f"Simplified: {res['latex']}", model=llm_model)
                st.markdown(expl)
            add_history(simp_input, "Simplify", res["latex"])
        except Exception as e:
            st.error(f"❌ {e}")

# ── WOLFRAM ALPHA ────────────────────────────────────────────────
elif mode == "🔭 Wolfram Alpha":
    st.markdown('<div class="math-card">', unsafe_allow_html=True)
    st.markdown("### 🔭 Wolfram Alpha  ·  Requires `WOLFRAM_APP_ID` in secrets.toml")
    wa_input = st.text_input("Query", placeholder="solve x^3 - 6x^2 + 11x - 6 = 0",
                              label_visibility="collapsed")
    wa_btn = st.button("🔭 Query Wolfram Alpha", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if wa_btn and wa_input.strip():
        with st.spinner("🔭 Querying..."):
            result = wolfram_query(wa_input)
        if "error" in result:
            st.error(f"Wolfram Alpha: {result['error']}")
            st.info("Get free API key at developer.wolframalpha.com")
        else:
            for title, content in result.items():
                st.markdown(f"""
                <div class="step-card">
                    <strong style="color:#f0c060">{title}</strong><br>
                    <pre style="color:#8b949e;font-size:.85rem;margin-top:6px;white-space:pre-wrap">{content}</pre>
                </div>""", unsafe_allow_html=True)
        add_history(wa_input, "Wolfram")

# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown('<div class="math-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;padding:16px 0;color:#484f58;font-size:.8rem;font-family:'JetBrains Mono',monospace">
    Built by <span style="color:#58a6ff">Muhammad Tahzeeb Shah</span> · BS Mathematics · AI Engineer · Pakistan 🇵🇰<br>
    <span style="font-size:.72rem">Powered by: SymPy CAS · Groq LLM · Plotly · Wolfram Alpha · Streamlit</span>
</div>
""", unsafe_allow_html=True)
