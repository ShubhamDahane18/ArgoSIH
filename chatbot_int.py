import os
import pandas as pd
from sqlalchemy import create_engine, text
from groq import Groq
import matplotlib.pyplot as plt

# ===== CONFIG =====
groq_api_key = os.getenv("GROQ_API_KEY")
engine = create_engine("postgresql+psycopg2://user:password@localhost:5432/Permalist")

# ===== FUNCTIONS =====
def load_argo_profiles(limit=1000):
    with engine.connect() as conn:
        result = conn.execute(text("SELECT * FROM argo_profiles LIMIT :lim"), {"lim": limit})
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df

def preprocess_df(df):
    for col in df.columns:
        if 'date' in col.lower():
            df[col] = pd.to_datetime(df[col], errors='coerce')
        elif df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col])
            except:
                pass
    return df

# ===== MAIN FLOW =====
df = load_argo_profiles()
df = preprocess_df(df)

if not groq_api_key:
    print(" Please provide your Groq API key as an environment variable.")
else:
    user_query = "How many profiles have Active = True?"  
    prompt = f"""
You are a Python data analyst. Given a pandas DataFrame named `df`, write Python code using pandas to answer this question:
Question: {user_query}
Only return Python code (no explanation). Use 'result' as the final output variable.
"""
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama-3.3-70b-versatile"
    )

    code_generated = chat_completion.choices[0].message.content.strip("`python").strip("`")
    local_vars = {"df": df}
    exec(code_generated, {}, local_vars)

    result = local_vars.get("result", " No result generated.")
    print("\n Final Result:\n", result)

    # Optional plot
    if isinstance(result, pd.DataFrame):
        result.plot(kind='bar')
        plt.show()
