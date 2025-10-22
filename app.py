import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTNeoForCausalLM
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
import openai
import os
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import uuid
import json
from flask import send_file, after_this_request
from datetime import datetime
import numpy as np
from collections import defaultdict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
matplotlib.use('Agg')
import gensim.downloader as api

app = Flask(__name__)
openai.api_key = 'your key'

USER_DATA_DIR = 'user_data'
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)

current_dir = os.path.dirname(os.path.abspath(__file__))

print("Loading Transformer models...")
models = {
    'gpt2': {
        'tokenizer': AutoTokenizer.from_pretrained('gpt2'),
        'model': AutoModelForCausalLM.from_pretrained('gpt2')
    },
    'gpt2-medium': {
        'tokenizer': AutoTokenizer.from_pretrained('gpt2-medium'),
        'model': AutoModelForCausalLM.from_pretrained('gpt2-medium')
    },
    'gpt-neo': {
        'tokenizer': AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-1.3B'),
        'model': GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-1.3B')
    }
}
print("Transformer models loaded.")

try:
    print("Loading word2vec-google-news-300 model...")
    word2vec_model = api.load('word2vec-google-news-300')
    models['word2vec-google-news-300'] = word2vec_model
    print("Word2Vec model loaded successfully.")
except Exception as e:
    print(f"!!!!!!!!!! FAILED TO LOAD WORD2VEC MODEL: {e} !!!!!!!!!!")
    print("The embedding visualizer will not work for this model.")

conversation_history = []


def save_user_interaction(user_id, interaction_type, input_text, model, temperature, temp_slider, p_value, seed,
                          predicted_tokens=None, predicted_probs=None, selected_word=None, selection_method="typed"):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            user_data = json.load(f)
    else:
        user_data = {"user_id": user_id, "interactions": []}
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "interaction_type": interaction_type,
        "selection_method": selection_method,
        "input_text": input_text,
        "model": model,
        "temperature": temperature,
        "chart_temperature": temp_slider,
        "p_value": p_value,
        "seed": seed,
        "selected_word": selected_word,
        "suggested_words": [{"word": word, "probability": prob} for word, prob in zip(predicted_tokens, predicted_probs)] if predicted_tokens else []
    }
    user_data["interactions"].append(interaction)
    with open(user_file, "w") as f:
        json.dump(user_data, f, indent=4)

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        first_name = request.form["first_name"].strip().capitalize()
        last_initial = request.form["last_initial"].strip().upper()
        grade = request.form["grade"].strip()
        q1= request.form.get("q1", "").strip()
        user_id = str(uuid.uuid4())
        user_data = {
            "user_id": user_id, "first_name": first_name, "last_initial": last_initial,
            "grade": grade, "q1": q1, "interactions": []
        }
        user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=4)
        return redirect(url_for("home", user_id=user_id))
    return render_template("login.html")

@app.route("/home")
def home():
    user_id = request.args.get("user_id")
    if not user_id: return redirect(url_for("login"))
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file): return redirect(url_for("login"))
    with open(user_file, "r") as f: user_data = json.load(f)
    return render_template("index.html", user_data=user_data, user_id=user_id)

@app.route('/index', methods=['GET'])
def index():
    user_file = request.args.get('user_file', 'default_user.json')
    return render_template('index.html', user_file=user_file)

@app.route("/chat")
def chat():
    user_id = request.args.get("user_id")
    if not user_id: return redirect(url_for("login"))
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if not os.path.exists(user_file): return redirect(url_for("login"))
    return render_template('chat.html', user_id=user_id)



@app.route('/save_interaction', methods=['POST'])
def save_interaction():
    data = request.json
    save_user_interaction(
        user_id=data.get('user_id', 'unknown_user'),
        interaction_type=data.get('interaction_type', 'unknown'),
        selection_method=data.get('selection_method', data.get('interaction_type')),
        input_text=data.get('input_text', ''),
        selected_word=data.get('selected_word'),
        model=data.get('model', 'unknown'),
        temperature=float(data.get('temperature', 1.0)),
        temp_slider=float(data.get('chart_temperature', 1.0)),
        p_value=float(data.get('p_value', 0.0)),
        seed=int(data.get('seed', 42)),
        predicted_tokens=data.get('predicted_tokens', []),
        predicted_probs=data.get('probabilities', [])
    )
    return jsonify({"status": "success"})

def clean_token(token):
    token = token.lstrip('Ġ').strip()
    return token if token.isalnum() else None

@app.route("/embedding")
def embedding_page():
    user_id = request.args.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    return render_template("embedding.html", user_id=user_id)

@app.route("/embedding", methods=["POST"])
def get_embedding():
    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id: return jsonify({"error": "User not authenticated"}), 401
    text = data.get("text", "").strip()
    model_name = data.get("model", "gpt2")
    if not text: return jsonify({"error": "Input text is required."}), 400
    if model_name not in models: return jsonify({"error": f"Model '{model_name}' not found."}), 404
    try:
        if model_name == 'word2vec-google-news-300':
            model = models[model_name]
            input_words = text.lower().split()
            words_to_plot = [word for word in input_words if word in model]
            if not words_to_plot or len(words_to_plot) < 2:
                return jsonify({"error": "Please enter at least 2 words found in the vocabulary."}), 400
            vectors_to_plot = np.array([model[w] for w in words_to_plot])
            perplexity_value = min(30.0, float(len(words_to_plot) - 1))
            reducer = TSNE(n_components=3, perplexity=perplexity_value, random_state=42)
            reduced_vectors = reducer.fit_transform(vectors_to_plot)
        else:
            tokenizer = models[model_name]["tokenizer"]
            model = models[model_name]["model"]
            input_ids = tokenizer.encode(text, return_tensors="pt")
            with torch.no_grad():
                embedding_layer = model.get_input_embeddings()
                vectors = embedding_layer(input_ids)[0].cpu().numpy()
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            token_vectors = defaultdict(list)
            for tok, vec in zip(tokens, vectors):
                cleaned_tok = tok.lstrip('Ġ')
                if cleaned_tok: token_vectors[cleaned_tok].append(vec)
            if not token_vectors: return jsonify({"error": "Could not find valid tokens."}), 400
            words_to_plot = list(token_vectors.keys())
            if len(words_to_plot) < 2: return jsonify({"error": "Please enter at least 2 unique words."}), 400
            averaged_vectors = np.array([np.mean(np.vstack(vecs), axis=0) for vecs in token_vectors.values()])
            perplexity_value = min(30.0, float(len(words_to_plot) - 1))
            reducer = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')
            reduced_vectors = reducer.fit_transform(averaged_vectors)
    except Exception as e:
        return jsonify({"error": f"Calculation failed: {str(e)}"}), 500
    points = [{"word": word, "x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])} for word, coords in zip(words_to_plot, reduced_vectors)]
    return jsonify({"points": points})

@app.route("/calculate_analogy", methods=["POST"])
def calculate_analogy():
    data = request.get_json()
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "User not authenticated"}), 401

    model_name = data.get("model", "gpt2")
    word_a_str, word_b_str, word_c_str = data.get("word_a"), data.get("word_b"), data.get("word_c")

    if not all([word_a_str, word_b_str, word_c_str]):
        return jsonify({"error": "Invalid input for analogy."}), 400

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found."}), 404

    try:

        # We need the vectors for all words involved to plot them.
        words_to_plot = [word_a_str, word_b_str, word_c_str]
        result_word = ""

        if model_name == 'word2vec-google-news-300':
            model = models[model_name]

            # Gensim's most_similar is efficient for finding the result
            results = model.most_similar(
                positive=[word_a_str.lower(), word_c_str.lower()],
                negative=[word_b_str.lower()],
                topn=1
            )
            result_word = results[0][0]
            words_to_plot.append(result_word)

            # Now get the vectors for the four words to be plotted
            vectors_to_plot = np.array([model[w.lower()] for w in words_to_plot])

        else: # It's a Transformer model
            tokenizer = models[model_name]["tokenizer"]
            model_obj = models[model_name]["model"]
            embedding_layer = model_obj.get_input_embeddings()

            def get_vector(word_str):
                token_ids = tokenizer.encode(word_str, return_tensors='pt')
                with torch.no_grad():
                    return embedding_layer(token_ids).mean(dim=1).cpu().numpy()

            vec_a, vec_b, vec_c = get_vector(word_a_str), get_vector(word_b_str), get_vector(word_c_str)
            result_vec = vec_a - vec_b + vec_c

            # Find the result word
            all_embeddings = embedding_layer.weight.detach().cpu().numpy()
            similarities = cosine_similarity(result_vec, all_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:10]

            input_words = {w.strip() for w in words_to_plot}
            for index in top_indices:
                word = tokenizer.decode(index).strip()
                if word and word not in input_words:
                    result_word = word
                    break

            words_to_plot.append(result_word)
            vec_d = get_vector(result_word)
            vectors_to_plot = np.vstack([vec_a, vec_b, vec_c, vec_d])

        # Perform dimensionality reduction on the four vectors
        # Perplexity must be less than n_samples, so for 4 points, it must be < 4.
        reducer = TSNE(n_components=3, perplexity=3.0, random_state=42, init='pca', learning_rate='auto')
        reduced_vectors = reducer.fit_transform(vectors_to_plot)

        points = [{"word": word, "x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])} for word, coords in zip(words_to_plot, reduced_vectors)]

        # Return BOTH the result word AND the points for the plot
        return jsonify({"result_word": result_word, "points": points})
        # --- END OF MODIFIED LOGIC ---

    except KeyError as e:
        return jsonify({"error": f"A word was not in the vocabulary: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Calculation failed: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
