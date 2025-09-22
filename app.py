from flask import Flask, request, render_template, jsonify
from vectorstore import save_to_postgres, build_vectorstore

app = Flask(__name__)

# Path to your CSV file
csv_path = r"E:\SIH 2025\ArgoSIH\indian_ocean_index.csv"

# Build the vectorstore directly from CSV
store = build_vectorstore(csv_path)

# Save the raw CSV data locally (mock for Postgres)
import pandas as pd
df = pd.read_csv(csv_path)
save_to_postgres(df, "argo_profiles", "demo")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form.get('query')
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    try:
        # Search the vectorstore
        results = store.similarity_search(user_query, k=3)
        return jsonify({"result": [r.page_content for r in results]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
