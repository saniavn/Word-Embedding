import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2Tokenizer, GPTNeoForCausalLM
import matplotlib.pyplot as plt
import io
import base64
import matplotlib
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


USER_DATA_DIR = 'user_data'
if not os.path.exists(USER_DATA_DIR):
    os.makedirs(USER_DATA_DIR)


try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
    print("Warning: __file__ not defined. Using current working directory.")


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


try:
    print("Loading glove-wiki-gigaword-100 model...")
    glove_model = api.load('glove-wiki-gigaword-100')
    models['glove-wiki-gigaword-100'] = glove_model
    print("GloVe model loaded successfully.")
except Exception as e:
    print(f"!!!!!!!!!! FAILED TO LOAD GLOVE MODEL: {e} !!!!!!!!!!")
    print("The embedding visualizer will not work for this model.")



conversation_history = []

GENSIM_MODELS = {'word2vec-google-news-300', 'glove-wiki-gigaword-100'}


def save_user_interaction(user_id, interaction_type, input_text, model, temperature, temp_slider, p_value, seed,
                          predicted_tokens=None, predicted_probs=None, selected_word=None, selection_method="typed"):
    user_file = os.path.join(USER_DATA_DIR, f"{user_id}.json")
    if os.path.exists(user_file):
        with open(user_file, "r") as f:
            try:
                user_data = json.load(f)
            except json.JSONDecodeError:
                user_data = {"user_id": user_id, "interactions": []} # Handle empty/corrupt file
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
    
    try:
        with open(user_file, "w") as f:
            json.dump(user_data, f, indent=4)
    except Exception as e:
        print(f"Error saving user interaction for {user_id}: {e}")

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        first_name = request.form["first_name"].strip().capitalize()
        last_initial = request.form["last_initial"].strip().upper()
        grade = request.form["grade"].strip()
        q2= request.form.get("q2", "").strip()
        user_id = str(uuid.uuid4())
        user_data = {
            "user_id": user_id, "first_name": first_name, "last_initial": last_initial,
            "grade": grade, "q2": q2, "interactions": []
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
    user_id = request.args.get('user_id')
    if user_id:
        return redirect(url_for("home", user_id=user_id))
    return redirect(url_for("login"))


@app.route('/save_interaction', methods=['POST'])
def save_interaction():
    data = request.json
    try:
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
    except Exception as e:
        print(f"Error in /save_interaction: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

def clean_token(token):
    token = token.lstrip('Ġ').strip()
    return token if token.isalnum() else None

@app.route("/embedding_page") # Renamed from /embedding to avoid conflict
def embedding_page():
    user_id = request.args.get("user_id")
    if not user_id:
        return redirect(url_for("login"))
    

    return redirect(url_for("home", user_id=user_id))


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
        if model_name in GENSIM_MODELS:
            model = models[model_name]
            input_words = text.lower().split()
            words_to_plot = [word for word in input_words if word in model]
            
            if not words_to_plot or len(words_to_plot) < 2:
                return jsonify({"error": "Please enter at least 2 words found in the vocabulary."}), 400
            
            vectors_to_plot = np.array([model[w] for w in words_to_plot])
            
            perplexity_value = min(30.0, float(len(words_to_plot) - 1)) 
            if perplexity_value < 1: perplexity_value = 1.0 # Ensure perplexity is at least 1
            
            reducer = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')
            reduced_vectors = reducer.fit_transform(vectors_to_plot)
        
        else: # It's a Transformer model
            tokenizer = models[model_name]["tokenizer"]
            model_obj = models[model_name]["model"]
            
            input_ids = tokenizer.encode(text, return_tensors="pt")
            with torch.no_grad():
                embedding_layer = model_obj.get_input_embeddings()
                vectors = embedding_layer(input_ids)[0].cpu().numpy()
            
            tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
            token_vectors = defaultdict(list)
            for tok, vec in zip(tokens, vectors):
                cleaned_tok = tok.lstrip('Ġ') # Handle GPT-2's 'Ġ' prefix
                if cleaned_tok: token_vectors[cleaned_tok].append(vec)
            
            if not token_vectors: return jsonify({"error": "Could not find valid tokens."}), 400
            
            words_to_plot = list(token_vectors.keys())
            if len(words_to_plot) < 2: return jsonify({"error": "Please enter at least 2 unique words."}), 400
            
            averaged_vectors = np.array([np.mean(np.vstack(vecs), axis=0) for vecs in token_vectors.values()])
            
            perplexity_value = min(30.0, float(len(words_to_plot) - 1))
            if perplexity_value < 1: perplexity_value = 1.0 # Ensure perplexity is at least 1

            reducer = TSNE(n_components=3, perplexity=perplexity_value, random_state=42, init='pca', learning_rate='auto')
            reduced_vectors = reducer.fit_transform(averaged_vectors)
            
    except Exception as e:
        print(f"Error in /embedding: {e}")
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
        words_to_plot = [word_a_str, word_b_str, word_c_str]
        result_word = ""

        if model_name in GENSIM_MODELS:
            model = models[model_name]

            # Gensim's most_similar is efficient for finding the result
            results = model.most_similar(
                positive=[word_a_str.lower(), word_c_str.lower()],
                negative=[word_b_str.lower()],
                topn=1
            )
            result_word = results[0][0]
            words_to_plot.append(result_word)

            # Ensure all words are in vocab before proceeding
            vectors_to_plot_list = []
            for w in words_to_plot:
                if w.lower() in model:
                    vectors_to_plot_list.append(model[w.lower()])
                else:
        
                    raise KeyError(f"Word '{w}' not in {model_name} vocabulary.")
            vectors_to_plot = np.array(vectors_to_plot_list)

        else: 
            tokenizer = models[model_name]["tokenizer"]
            model_obj = models[model_name]["model"]
            embedding_layer = model_obj.get_input_embeddings()

            def get_vector(word_str):
                # Ensure to handle subwords by averaging
                token_ids = tokenizer.encode(word_str, return_tensors='pt')
                with torch.no_grad():
                    return embedding_layer(token_ids).mean(dim=1).cpu().numpy()

            vec_a, vec_b, vec_c = get_vector(word_a_str), get_vector(word_b_str), get_vector(word_c_str)
            result_vec = vec_a - vec_b + vec_c

            # Find the result word
            all_embeddings = embedding_layer.weight.detach().cpu().numpy()
            similarities = cosine_similarity(result_vec, all_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:20] # Check more indices

            input_words = {w.strip().lower() for w in words_to_plot}
            result_word = "" # Reset
            for index in top_indices:
                word = tokenizer.decode(index).strip()
                if word and word.lower() not in input_words and word.isalnum():
                    result_word = word
                    break
            
            if not result_word: # Fallback if no good word is found
                result_word = tokenizer.decode(top_indices[0]).strip()

            words_to_plot.append(result_word)
            vec_d = get_vector(result_word)
            vectors_to_plot = np.vstack([vec_a, vec_b, vec_c, vec_d])

        
        reducer = TSNE(n_components=3, perplexity=3.0, random_state=42, init='pca', learning_rate='auto')
        reduced_vectors = reducer.fit_transform(vectors_to_plot)

        points = [{"word": word, "x": float(coords[0]), "y": float(coords[1]), "z": float(coords[2])} for word, coords in zip(words_to_plot, reduced_vectors)]

        return jsonify({"result_word": result_word, "points": points})

    except KeyError as e:
        return jsonify({"error": f"A word was not in the vocabulary: {e}"}), 400
    except Exception as e:
        print(f"Error in /calculate_analogy: {e}")
        return jsonify({"error": f"Calculation failed: {e}"}), 500


@app.route('/download_user_data/<user_id>')
def download_user_data(user_id):
    # Sanitize user_id to prevent directory traversal
    safe_user_id = str(uuid.UUID(user_id)) # Ensures it's a valid UUID
    user_file_path = os.path.join(USER_DATA_DIR, f"{safe_user_id}.json")
    
    if not os.path.exists(user_file_path):
        return "File not found or invalid user ID.", 404
 

    
    return send_file(user_file_path, as_attachment=True, download_name=f"{safe_user_id}_data.json")



if __name__ == '__main__':
    app.run(debug=True)
