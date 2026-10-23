"""SmartTopic_LDA - Advanced NLP Topic Modeling Web Application.

Built for Topic Modeling coursework demonstration using Latent Dirichlet Allocation (LDA),
scikit-learn, NLTK, Plotly, and Streamlit.
"""

import io
import re
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
import streamlit as st

# Safe NLTK stopwords and lemmatizer loading
try:
    import nltk
    from nltk.corpus import stopwords as nltk_stopwords
    from nltk.stem import WordNetLemmatizer

    try:
        nltk.data.find("corpora/stopwords")
    except (LookupError, AttributeError):
        nltk.download("stopwords", quiet=True)

    try:
        nltk.data.find("corpora/wordnet")
    except (LookupError, AttributeError):
        nltk.download("wordnet", quiet=True)

    try:
        NLTK_STOP = set(nltk_stopwords.words("english"))
    except Exception:
        NLTK_STOP = set()

    try:
        LEMMATIZER = WordNetLemmatizer()
    except Exception:
        LEMMATIZER = None
except Exception:
    NLTK_STOP = set()
    LEMMATIZER = None

BASE_STOPWORDS = set(ENGLISH_STOP_WORDS).union(NLTK_STOP)

# Page Setup
st.set_page_config(
    page_title="SmartTopic_LDA | Topic Modeling",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional CSS Styling
st.markdown(
    """
    <style>
    .viva-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.15rem;
    }
    .viva-sub {
        font-size: 1.0rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .metric-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .topic-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .topic-pill {
        display: inline-block;
        padding: 3px 8px;
        background-color: #2563EB;
        color: white;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.8rem;
        margin-bottom: 6px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Built-in Sample Datasets for College Viva
SAMPLE_DATASETS = {
    "AI & Computer Science Research (30 docs)": [
        # Sub-topic: NLP & Large Language Models
        "Transformer architectures utilize self-attention mechanisms to model contextual dependencies across sequential tokens in text.",
        "Large language models pre-trained on massive text corpora demonstrate zero-shot generalization and reasoning across complex benchmarks.",
        "Prompt engineering and chain-of-thought prompting improve multi-step logical deduction in generative language models.",
        "Tokenization algorithms like byte-pair encoding break input text into subword vocabularies for language modeling.",
        "Fine-tuning pretrained transformer models on domain-specific datasets yields high performance on classification tasks.",
        "Masked language modeling trains neural networks to predict hidden tokens using bidirectional contextual representations.",
        "Vector embeddings map words and semantic concepts into continuous low-dimensional geometric vector spaces.",
        "Retrieval-augmented generation grounds language model responses in verified external knowledge databases.",
        # Sub-topic: Cloud & Distributed Systems
        "Kubernetes manages containerized microservices across distributed server clusters with automated load balancing and scaling.",
        "Docker containers package application code, system libraries, and runtime dependencies into isolated lightweight images.",
        "Microservice architectures decouple monolithic software into modular independently deployable backend network services.",
        "Distributed message queues such as Apache Kafka provide high-throughput publish-subscribe streaming data pipelines.",
        "Infrastructure as code tools like Terraform automate cloud server provisioning, firewall rules, and virtual networks.",
        "Database sharding and horizontal read replication partition high-volume transactions across multiple database nodes.",
        "API gateways handle reverse proxy routing, rate limiting, and client token authentication for backend services.",
        # Sub-topic: Cybersecurity & Encryption
        "Zero trust network security architecture mandates continuous verification of user identity and endpoint security posture.",
        "Public key cryptography uses asymmetric key pairs with mathematical trapdoor functions for secure digital signatures.",
        "Penetration testing and vulnerability scanning identify software flaws like SQL injection and cross-site scripting.",
        "Ransomware attacks encrypt critical enterprise storage filesystems and demand cryptocurrency payments for decryption keys.",
        "Multi-factor authentication enforces biometric verification and hardware tokens to prevent credential theft.",
        "Web application firewalls inspect incoming HTTP network payloads to block distributed denial-of-service traffic.",
        "Secure socket layer and transport layer security encrypt network communications between web browsers and servers.",
        # Sub-topic: Quantum Computing
        "Superconducting quantum circuits operate at millikelvin dilution refrigerator temperatures to preserve qubit coherence.",
        "Quantum entanglement creates non-local physical correlations between qubits for quantum teleportation protocols.",
        "Shor's quantum algorithm achieves polynomial time prime factorization, challenging classical RSA cryptography.",
        "Quantum error correction codes protect logical qubits against environmental decoherence and bit-flip phase noise.",
        "Variational quantum eigensolver algorithms simulate molecular ground state energy configurations on noisy quantum devices.",
        "Quantum key distribution utilizes single photon polarization states to establish secure cryptographic keys.",
        "Grover's quantum search algorithm provides quadratic speedup for searching unstructured database elements.",
        "Quantum supremacy experiments demonstrate computational tasks executed faster on quantum hardware than supercomputers.",
    ],
    "Biomedical & Healthcare Science (30 docs)": [
        # Sub-topic: Cardiology & Cardiovascular Health
        "Atherosclerosis develops when low-density lipoprotein cholesterol plaques accumulate within coronary arterial walls.",
        "Coronary artery disease reduces oxygenated blood perfusion to cardiac myocardium, precipitating acute myocardial infarction.",
        "Electrocardiograms detect lethal cardiac arrhythmias including ventricular fibrillation and atrial flutter.",
        "Systemic arterial hypertension increases cardiac afterload and serves as a major risk factor for congestive heart failure.",
        "Echocardiography assesses left ventricular ejection fraction, myocardial wall thickness, and heart valve regurgitation.",
        "Aerobic endurance exercise enhances myocardial stroke volume and stimulates collateral coronary angiogenesis.",
        "Statins inhibit HMG-CoA reductase to lower circulating low-density lipoprotein cholesterol in cardiac patients.",
        # Sub-topic: Immunology & Vaccines
        "Messenger RNA vaccines deliver lipid nanoparticles encoding viral spike proteins to stimulate neutralizing antibodies.",
        "Adaptive immune responses rely on memory B lymphocytes to synthesize pathogen-specific protective antibodies.",
        "Cytotoxic CD8 T lymphocytes recognize and destroy virally infected host cells presenting foreign peptides on MHC-I.",
        "Autoimmune diseases occur when self-tolerance breaks down and the immune system targets endogenous healthy tissues.",
        "Adjuvants in vaccine formulations boost innate immune receptor signaling and enhance long-term immunological memory.",
        "Monoclonal antibody therapeutics bind and neutralize specific pro-inflammatory cytokines during severe systemic inflammation.",
        "Phagocytic macrophages and neutrophils engulf invading bacterial microorganisms through receptor-mediated endocytosis.",
        # Sub-topic: Neuroscience & Brain Health
        "Synaptic plasticity and long-term potentiation constitute the cellular neurobiological basis of memory formation.",
        "Neurotransmitters including dopamine, serotonin, and norepinephrine regulate cognitive motivation, mood, and alertness.",
        "Alzheimer's disease pathology involves neurotoxic extracellular amyloid-beta plaques and intraneuronal tau protein tangles.",
        "The blood-brain barrier protects central nervous system parenchyma from circulating peripheral toxins and pathogens.",
        "Functional magnetic resonance imaging monitors regional cerebral blood flow during cognitive sensory processing tasks.",
        "Slow-wave non-REM sleep promotes glymphatic clearance of metabolic waste products from central brain parenchyma.",
        "Dendritic spine remodeling in hippocampal subfields facilitates spatial learning and adaptive emotional conditioning.",
        # Sub-topic: Nutrition & Metabolic Health
        "The gastrointestinal gut microbiome ferments indigestible dietary fibers into beneficial short-chain fatty acids.",
        "Peripheral insulin resistance in skeletal muscle and hepatic tissues leads to hyperglycemia and type 2 diabetes.",
        "Omega-3 polyunsaturated fatty acids reduce chronic systemic inflammatory markers and improve lipid profiles.",
        "Intermittent fasting promotes cellular autophagy and stimulates mitochondrial biogenesis in hepatic metabolic tissues.",
        "Probiotic dietary supplementation with Lactobacillus and Bifidobacterium strains enhances intestinal mucosal barrier integrity.",
        "Excess consumption of refined carbohydrates and fructose accelerates hepatic steatosis and non-alcoholic fatty liver disease.",
        "Dietary micronutrients such as vitamin D and magnesium act as essential cofactors in hundreds of metabolic enzyme reactions.",
        "Metabolic syndrome is clinically characterized by central obesity, dyslipidemia, hypertension, and impaired glucose tolerance.",
        "The bidirectional gut-brain axis coordinates neuroendocrine signaling between intestinal microbiota and the nervous system.",
    ],
    "Customer Support & Product Reviews (30 docs)": [
        # Sub-topic: Billing & Subscriptions
        "My credit card was charged twice for the monthly subscription renewal without my authorization. Please refund immediately.",
        "How can I update my billing payment method and credit card expiration date before the next invoice cycle?",
        "Our accounting team requires official PDF tax invoices for the past three quarterly billing statements.",
        "My subscription was canceled unexpectedly even though my payment card is active and has sufficient funds.",
        "Can you explain the unexpected overage charges that appeared on our latest monthly enterprise usage bill?",
        "Please cancel my recurring subscription renewal and delete my payment details from your customer database.",
        "I downgraded my team subscription plan last week but have not received the promised prorated account refund.",
        "Where can I find the company tax ID and formal billing receipts for corporate expense reimbursement?",
        # Sub-topic: Authentication & Account Access
        "I forgot my account login password and the password reset link email is never delivered to my inbox.",
        "I lost access to my authenticator device and cannot complete the two-factor authentication verification step.",
        "Someone attempted an unauthorized login to my profile from an unrecognized IP address. Please secure my account.",
        "Single sign-on via Google workspace fails with an authentication token expired error during login.",
        "My account has been locked due to multiple incorrect password attempts. Please help unlock my profile.",
        "Can an administrator transfer account owner permissions to another registered team member email?",
        "I need backup emergency recovery codes because my mobile phone with the authenticator app was damaged.",
        # Sub-topic: Shipping & Package Delivery
        "The courier tracking status shows delivered but I have checked my front porch and no package was left.",
        "My order shipment has been stuck in transit at the regional sorting distribution hub for ten days.",
        "Can I update the delivery shipping address for my parcel before the warehouse dispatches the order?",
        "The courier attempted delivery without ringing the doorbell and left a missed delivery card on my gate.",
        "Do you offer expedited next-day express shipping for residential addresses in suburban areas?",
        "The package tracking link provided in the confirmation email displays an invalid tracking number error.",
        "The delivery parcel arrived completely crushed and opened with several ordered items missing from the carton.",
        # Sub-topic: Returns, Exchanges & Hardware
        "I would like to return the footwear because the size is too small and exchange it for a larger size.",
        "Please email me a prepaid return shipping postage label so I can send back the defective electronic item.",
        "What is your return policy return window? Is it thirty days from product delivery or from invoice date?",
        "I dropped off the return package at the drop-off location five days ago. When will my refund be processed?",
        "The replacement hardware unit sent by your warranty department exhibits the exact same power failure bug.",
        "Can I exchange this gift purchase for store credit without possessing the original physical paper receipt?",
        "Customer support promised a replacement unit under warranty but I have not received a dispatch tracking code.",
    ],
}


# Preprocessing Helper Function
def preprocess_text(
    text: str,
    apply_lemmatization: bool = True,
    remove_numbers: bool = True,
    min_word_len: int = 3,
    custom_stopwords: set = None,
) -> str:
    """Clean and preprocess an individual text string."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    if remove_numbers:
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    tokens = text.split()

    active_stopwords = BASE_STOPWORDS.union(custom_stopwords or set())

    cleaned_tokens = []
    for token in tokens:
        if len(token) < min_word_len:
            continue
        if token in active_stopwords:
            continue
        if apply_lemmatization and LEMMATIZER is not None:
            try:
                token = LEMMATIZER.lemmatize(token)
            except Exception:
                pass
        if token not in active_stopwords and len(token) >= min_word_len:
            cleaned_tokens.append(token)

    return " ".join(cleaned_tokens)


# Header
st.markdown("<div class='viva-header'>🧠 SmartTopic_LDA</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='viva-sub'>Advanced Natural Language Processing — Unsupervised Topic Modeling with Scikit-Learn Latent Dirichlet Allocation (LDA)</div>",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.header("📂 1. Corpus Input")
    data_mode = st.radio(
        "Corpus Source:",
        ["Built-in Sample Datasets", "Upload Custom CSV File"],
        index=0,
    )

    documents: List[str] = []

    if data_mode == "Built-in Sample Datasets":
        dataset_key = st.selectbox("Select Sample Corpus:", list(SAMPLE_DATASETS.keys()), index=0)
        documents = SAMPLE_DATASETS[dataset_key]
        st.success(f"Loaded **{len(documents)}** sample documents.")
    else:
        uploaded_file = st.file_uploader("Upload CSV file:", type=["csv"])
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                st.write(f"CSV uploaded: {df_upload.shape[0]} rows, {df_upload.shape[1]} columns")

                text_column = st.selectbox("Select Text Column:", df_upload.columns.tolist(), index=0)
                raw_texts = df_upload[text_column].dropna().astype(str).tolist()
                documents = [t.strip() for t in raw_texts if t.strip()]

                if documents:
                    st.success(f"Extracted **{len(documents)}** non-empty text rows.")
                    with st.expander("👁️ Preview raw text"):
                        st.dataframe(df_upload[[text_column]].head(5))
                else:
                    st.error("Selected column contains no valid non-empty text.")
            except Exception as e:
                st.error(f"Error reading CSV file: {e}")

    st.markdown("---")
    st.header("🧹 2. NLP Preprocessing")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        lemmatize_opt = st.checkbox("Lemmatize", value=True, help="Reduce tokens to base form (e.g. models -> model)")
    with col_p2:
        strip_nums_opt = st.checkbox("Strip Digits", value=True)

    min_len = st.slider("Min Word Length", min_value=2, max_value=5, value=3)

    ngram_option = st.selectbox(
        "N-gram Selection:",
        ["Unigrams only (1, 1)", "Unigrams + Bigrams (1, 2)"],
        index=1,
    )
    ngram_range = (1, 1) if ngram_option.startswith("Unigrams only") else (1, 2)

    custom_stop_input = st.text_input(
        "Custom Stopwords (comma separated):",
        placeholder="e.g. also, said, use, one",
        help="Words to filter out of the vocabulary.",
    )
    custom_stops = set(w.strip().lower() for w in custom_stop_input.split(",") if w.strip())

    st.markdown("---")
    st.header("⚙️ 3. LDA Model Hyperparameters")
    n_topics = st.slider("Number of Topics (K)", min_value=2, max_value=8, value=4, help="Target number of latent topics")
    top_words_count = st.slider("Top Words per Topic", min_value=5, max_value=15, value=10)
    max_iter = st.slider("Max Iterations", min_value=10, max_value=50, value=25, step=5)
    random_state = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0, max_value=9999)

    run_btn = st.button("🚀 Train LDA Topic Model", type="primary", use_container_width=True)


# Processing & Training Logic
if run_btn:
    # 1. Validation for empty text
    if not documents or len(documents) < 2:
        st.error("❌ Error: The corpus contains insufficient documents. Please provide at least 2 text documents.")
        st.stop()

    with st.spinner("Executing NLP Preprocessing and Document-Term Matrix (DTM) construction..."):
        cleaned_docs = [
            preprocess_text(
                doc,
                apply_lemmatization=lemmatize_opt,
                remove_numbers=strip_nums_opt,
                min_word_len=min_len,
                custom_stopwords=custom_stops,
            )
            for doc in documents
        ]

        # Filter out empty documents after cleaning
        valid_pairs = [(orig, clean) for orig, clean in zip(documents, cleaned_docs) if clean.strip()]

        if not valid_pairs:
            st.error(
                "❌ Error: Preprocessing removed all words across all documents! "
                "Check your minimum word length or custom stopwords."
            )
            st.stop()

        active_raw_docs = [p[0] for p in valid_pairs]
        active_clean_docs = [p[1] for p in valid_pairs]

        try:
            vectorizer = CountVectorizer(
                ngram_range=ngram_range,
                min_df=1,
                max_df=0.98,
                token_pattern=r"(?u)\b\w+\b",
            )
            dtm = vectorizer.fit_transform(active_clean_docs)
            feature_names = vectorizer.get_feature_names_out()
        except Exception as e:
            st.error(f"❌ Error during Document-Term Matrix construction: {e}")
            st.stop()

        # 2. Validation for insufficient vocabulary
        vocab_size = dtm.shape[1]
        if vocab_size < n_topics:
            st.error(
                f"❌ Error: Insufficient vocabulary! The extracted vocabulary has only {vocab_size} unique terms, "
                f"which is less than the requested topic count K = {n_topics}. "
                f"Please reduce K or provide richer documents."
            )
            st.stop()

    with st.spinner(f"Fitting Scikit-Learn Latent Dirichlet Allocation (K={n_topics})..."):
        try:
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=max_iter,
                learning_method="batch",
                random_state=int(random_state),
                n_jobs=-1,
            )
            lda_model.fit(dtm)

            # Posterior topic distributions for documents
            doc_topic_probs = lda_model.transform(dtm)
            dominant_topics = np.argmax(doc_topic_probs, axis=1)
            confidences = np.max(doc_topic_probs, axis=1)

            # Normalized topic-word distributions (components_ sum to 1 per topic)
            topic_word_dists = []
            suggested_labels = {}
            topic_top_words_dict = {}

            for t_idx, comp in enumerate(lda_model.components_):
                norm_comp = comp / comp.sum() if comp.sum() > 0 else comp
                topic_word_dists.append(norm_comp)

                top_indices = norm_comp.argsort()[: -top_words_count - 1 : -1]
                top_words_weights = [(feature_names[i], float(norm_comp[i])) for i in top_indices]
                topic_top_words_dict[t_idx] = top_words_weights

                # Suggested label from top 3 keywords
                top3 = [w for w, _ in top_words_weights[:3]]
                suggested_labels[t_idx] = f"Topic {t_idx + 1}: {', '.join(top3)}"

            # Perplexity and Log Likelihood metrics
            perplexity = float(lda_model.perplexity(dtm))
            log_likelihood = float(lda_model.score(dtm))

            # Store in session state for instant multi-tab browsing
            st.session_state["lda_results"] = {
                "model": lda_model,
                "vectorizer": vectorizer,
                "dtm": dtm,
                "feature_names": feature_names,
                "raw_docs": active_raw_docs,
                "clean_docs": active_clean_docs,
                "doc_topic_probs": doc_topic_probs,
                "dominant_topics": dominant_topics,
                "confidences": confidences,
                "topic_word_dists": topic_word_dists,
                "suggested_labels": suggested_labels,
                "topic_top_words_dict": topic_top_words_dict,
                "n_topics": n_topics,
                "top_words_count": top_words_count,
                "perplexity": perplexity,
                "log_likelihood": log_likelihood,
                "vocab_size": vocab_size,
            }
            st.toast("✅ LDA Model Trained Successfully!", icon="🎉")
        except Exception as e:
            st.error(f"❌ Error during LDA model fitting: {e}")
            st.stop()


# Display Results if available
if "lda_results" in st.session_state:
    res = st.session_state["lda_results"]

    # Top Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Total Documents", len(res["raw_docs"]))
    with m2:
        st.metric("Vocabulary (V)", res["vocab_size"])
    with m3:
        st.metric("Topics (K)", res["n_topics"])
    with m4:
        st.metric("Model Perplexity", f"{res['perplexity']:.1f}", help="Perplexity evaluates language model generalization (lower is better).")
    with m5:
        st.metric("Log-Likelihood", f"{res['log_likelihood']:.0f}", help="Total log likelihood of observing the document-term matrix given the LDA parameters (higher is better).")

    st.markdown("<br>", unsafe_allow_html=True)

    # 5 Tabbed Sections for Viva Demonstration
    tab_topics, tab_vis, tab_docs, tab_report, tab_export = st.tabs([
        "🏷️ 1. Discovered Topics & Keywords",
        "📊 2. Interactive Plotly Visualizations",
        "📑 3. Document-Topic Probabilities",
        "📋 4. Viva Interpretation Report",
        "💾 5. Download Results & Artifacts",
    ])

    # Color Palette for Topics
    VIVA_COLORS = ["#2563EB", "#7C3AED", "#DB2777", "#059669", "#D97706", "#0891B2", "#DC2626", "#4F46E5"]

    # ---------------- TAB 1: Topics & Keywords ----------------
    with tab_topics:
        st.subheader("Discovered Latent Topics & Suggested Labels")
        st.markdown(
            "Below are the **$K$ latent topics** uncovered by the Scikit-Learn LDA Dirichlet-Multinomial model. "
            "You can review the suggested labels, top words, and their exact normalized conditional probabilities $P(w \\mid z = k)$."
        )

        cols = st.columns(min(3, res["n_topics"]))
        for t_idx in range(res["n_topics"]):
            col_target = cols[t_idx % len(cols)]
            with col_target:
                st.markdown("<div class='topic-card'>", unsafe_allow_html=True)
                st.markdown(f"<span class='topic-pill'>Topic #{t_idx + 1}</span>", unsafe_allow_html=True)

                current_label = res["suggested_labels"].get(t_idx, f"Topic {t_idx + 1}")
                user_label = st.text_input(
                    f"Label for Topic #{t_idx + 1}:",
                    value=current_label,
                    key=f"user_topic_label_{t_idx}",
                )
                res["suggested_labels"][t_idx] = user_label

                words_weights = res["topic_top_words_dict"][t_idx]
                doc_count = int(np.sum(res["dominant_topics"] == t_idx))
                pct = (doc_count / len(res["raw_docs"])) * 100

                st.markdown(f"**Dominant in:** `{doc_count}` docs ({pct:.1f}%)")
                st.markdown("**Top Keywords & Weights:**")

                df_kw = pd.DataFrame(words_weights, columns=["Term", "P(w|Topic)"])
                df_kw["P(w|Topic)"] = df_kw["P(w|Topic)"].apply(lambda x: f"{x:.4f}")
                st.dataframe(df_kw.head(8), hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- TAB 2: Visualizations ----------------
    with tab_vis:
        st.subheader("Interactive Plotly Topic Visualizations")

        vis_sub1, vis_sub2, vis_sub3, vis_sub4 = st.tabs([
            "📈 Topic-Word Importance Bars",
            "🍩 Topic Prevalence Donut",
            "🗺️ Intertopic Distance Map (2D PCA)",
            "🔥 Document Mixture Heatmap",
        ])

        # Sub 1: Faceted Word Importance Bar Charts
        with vis_sub1:
            n_t = res["n_topics"]
            ncols = 2 if n_t > 1 else 1
            nrows = (n_t + 1) // 2 if ncols == 2 else n_t

            fig_bars = make_subplots(
                rows=nrows,
                cols=ncols,
                subplot_titles=[res["suggested_labels"][i] for i in range(n_t)],
                horizontal_spacing=0.15,
                vertical_spacing=0.14 if nrows > 2 else 0.20,
            )

            for i in range(n_t):
                r = (i // ncols) + 1
                c = (i % ncols) + 1
                w_weights = res["topic_top_words_dict"][i][: res["top_words_count"]]
                w_list = [w for w, _ in reversed(w_weights)]
                v_list = [wt for _, wt in reversed(w_weights)]
                color = VIVA_COLORS[i % len(VIVA_COLORS)]

                fig_bars.add_trace(
                    go.Bar(
                        x=v_list,
                        y=w_list,
                        orientation="h",
                        marker=dict(color=color),
                        name=f"Topic {i + 1}",
                        hovertemplate="<b>%{y}</b><br>Probability: %{x:.4f}<extra></extra>",
                        showlegend=False,
                    ),
                    row=r,
                    col=c,
                )

            fig_bars.update_layout(
                height=max(420, nrows * 260),
                template="plotly_white",
                margin=dict(l=30, r=30, t=50, b=40),
                font=dict(family="sans-serif", size=11),
            )
            fig_bars.update_xaxes(title_text="Normalized Probability P(w|Topic)")
            st.plotly_chart(fig_bars, use_container_width=True)

        # Sub 2: Topic Prevalence Donut
        with vis_sub2:
            counts = [int(np.sum(res["dominant_topics"] == i)) for i in range(res["n_topics"])]
            labels = [res["suggested_labels"][i] for i in range(res["n_topics"])]

            fig_donut = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=counts,
                        hole=0.45,
                        marker=dict(colors=VIVA_COLORS[: len(counts)]),
                        textinfo="percent+label",
                        hovertemplate="<b>%{label}</b><br>Assigned Docs: %{value} (%{percent})<extra></extra>",
                    )
                ]
            )
            fig_donut.update_layout(
                title="<b>Corpus Document Distribution Across Topics</b>",
                template="plotly_white",
                height=450,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            )
            st.plotly_chart(fig_donut, use_container_width=True)

        # Sub 3: 2D Intertopic Distance Map
        with vis_sub3:
            st.markdown(
                "Projects the topic-word distributions into 2D space using **Principal Component Analysis (PCA)**. "
                "Bubble sizes correspond to document counts. Separation between bubbles indicates semantic distinction."
            )
            comp_matrix = np.array(res["topic_word_dists"])
            if res["n_topics"] >= 2:
                pca = PCA(n_components=2, random_state=42)
                coords = pca.fit_transform(comp_matrix)
                pc1, pc2 = coords[:, 0], coords[:, 1]
                var1, var2 = pca.explained_variance_ratio_[:2]
            else:
                pc1, pc2 = [0.0], [0.0]
                var1, var2 = 0.0, 0.0

            sizes = [max(25, counts[i] * 8) for i in range(res["n_topics"])]

            fig_pca = go.Figure()
            fig_pca.add_trace(
                go.Scatter(
                    x=pc1,
                    y=pc2,
                    mode="markers+text",
                    text=[f"T{i + 1}" for i in range(res["n_topics"])],
                    textposition="middle center",
                    textfont=dict(color="#ffffff", size=13, family="sans-serif"),
                    marker=dict(
                        size=sizes,
                        color=VIVA_COLORS[: res["n_topics"]],
                        opacity=0.88,
                        line=dict(width=2, color="#ffffff"),
                    ),
                    hoverinfo="text",
                    hovertext=[
                        f"<b>{labels[i]}</b><br>Docs: {counts[i]} ({counts[i] / len(res['raw_docs']) * 100:.1f}%)"
                        for i in range(res["n_topics"])
                    ],
                )
            )
            fig_pca.update_layout(
                title=f"<b>Intertopic Semantic Distance Map</b> (PC1: {var1 * 100:.1f}%, PC2: {var2 * 100:.1f}%)",
                xaxis_title="Principal Component 1",
                yaxis_title="Principal Component 2",
                template="plotly_white",
                height=480,
            )
            st.plotly_chart(fig_pca, use_container_width=True)

        # Sub 4: Document Mixture Heatmap
        with vis_sub4:
            sample_docs_count = min(30, len(res["raw_docs"]))
            st.markdown(f"Displays the **document-topic mixture matrix $\\theta_{{d, k}}$** for the first {sample_docs_count} documents.")

            sample_mat = res["doc_topic_probs"][:sample_docs_count]
            t_headers = [f"T{i + 1}" for i in range(res["n_topics"])]
            d_headers = [f"Doc {i + 1}" for i in range(sample_docs_count)]

            fig_hm = go.Figure(
                data=go.Heatmap(
                    z=sample_mat,
                    x=t_headers,
                    y=d_headers,
                    colorscale="Blues",
                    colorbar=dict(title="P(Topic|Doc)"),
                    hovertemplate="Document: %{y}<br>Topic: %{x}<br>Probability: %{z:.3f}<extra></extra>",
                )
            )
            fig_hm.update_layout(
                title="<b>Document-Topic Posterior Probabilities Matrix</b>",
                xaxis_title="Topics",
                yaxis_title="Documents",
                template="plotly_white",
                height=max(420, sample_docs_count * 20),
            )
            st.plotly_chart(fig_hm, use_container_width=True)

    # ---------------- TAB 3: Document Probabilities ----------------
    with tab_docs:
        st.subheader("📑 Document-Topic Distributions & Assignments")

        # Build comprehensive dataframe
        df_docs_data = {
            "Doc ID": list(range(1, len(res["raw_docs"]) + 1)),
            "Dominant Topic": [res["suggested_labels"][t] for t in res["dominant_topics"]],
            "Confidence": [round(float(c), 4) for c in res["confidences"]],
            "Document Snippet": [d[:120] + ("..." if len(d) > 120 else "") for d in res["raw_docs"]],
            "Cleaned Tokens": res["clean_docs"],
        }
        for t_idx in range(res["n_topics"]):
            df_docs_data[f"P(T{t_idx + 1})"] = [round(float(p), 4) for p in res["doc_topic_probs"][:, t_idx]]

        df_assignments = pd.DataFrame(df_docs_data)

        # Interactive filter
        t_filter = st.selectbox(
            "Filter by Dominant Topic:",
            ["All Topics"] + [res["suggested_labels"][i] for i in range(res["n_topics"])],
        )
        filtered_view = df_assignments.copy()
        if t_filter != "All Topics":
            filtered_view = filtered_view[filtered_view["Dominant Topic"] == t_filter]

        st.caption(f"Displaying **{len(filtered_view)}** of {len(df_assignments)} documents")

        st.dataframe(
            filtered_view[["Doc ID", "Dominant Topic", "Confidence", "Document Snippet"] + [f"P(T{i + 1})" for i in range(res["n_topics"])]],
            use_container_width=True,
            column_config={
                "Confidence": st.column_config.ProgressColumn(
                    "Confidence",
                    format="%.2f",
                    min_value=0.0,
                    max_value=1.0,
                ),
            },
            hide_index=True,
        )

    # ---------------- TAB 4: Viva Interpretation Report ----------------
    with tab_report:
        st.subheader("📋 Viva Interpretation Report")
        st.markdown(
            "This automated academic interpretation report provides comprehensive analysis for presentation during a viva examination."
        )

        report_md = f"""### SmartTopic_LDA Model Evaluation & Topic Interpretation Report

#### 1. Executive Summary
- **Total Ingested Documents ($D$):** {len(res['raw_docs'])}
- **Extracted Vocabulary Size ($V$):** {res['vocab_size']} unique terms
- **Configured Number of Topics ($K$):** {res['n_topics']}
- **Model Perplexity Score:** `{res['perplexity']:.2f}` *(Lower perplexity indicates higher predictive generalization on text)*
- **Model Total Log-Likelihood:** `{res['log_likelihood']:.2f}` *(Higher log-likelihood indicates superior parameter convergence)*

---

#### 2. Discovered Latent Topics & Semantic Interpretation
"""

        for t_idx in range(res["n_topics"]):
            label = res["suggested_labels"][t_idx]
            w_w = res["topic_top_words_dict"][t_idx]
            doc_cnt = int(np.sum(res["dominant_topics"] == t_idx))
            pct = (doc_cnt / len(res["raw_docs"])) * 100
            kw_str = ", ".join([f"**{w}** ({wt:.3f})" for w, wt in w_w[:6]])

            report_md += f"""
##### Topic {t_idx + 1}: {label}
- **Prevalence in Corpus:** `{doc_cnt}` documents ({pct:.1f}% of total dataset)
- **Representative Keywords:** {kw_str}
- **Interpretation:** This latent topic captures terms primarily characterizing domain semantics around *{", ".join([w for w, _ in w_w[:4]])}*. Documents assigned to this topic display strong semantic co-occurrence of these keywords.
"""

        report_md += """
---

#### 3. Theoretical Underpinnings (Latent Dirichlet Allocation)
- **Generative Process:** LDA is a three-level hierarchical Bayesian model that assumes each document is generated as a random mixture over latent topics, where each topic is a discrete probability distribution over the vocabulary.
- **Dirichlet Priors:** Document-topic distributions $\\theta_d \\sim \\text{Dir}(\\alpha)$ and topic-word distributions $\\phi_k \\sim \\text{Dir}(\\beta)$.
- **Inference Mechanism:** Scikit-Learn's `LatentDirichletAllocation` uses Variational Expectation-Maximization (Variational EM) to approximate the posterior distributions of $\\theta$ and $\\phi$.
"""
        st.markdown(report_md)

    # ---------------- TAB 5: Download Artifacts ----------------
    with tab_export:
        st.subheader("💾 Download Topic Modeling Results & Artifacts")
        st.markdown("Download all topic assignments, keyword distributions, and the interpretation report for offline evaluation.")

        exp_col1, exp_col2, exp_col3 = st.columns(3)

        # 1. Download Document Assignments CSV
        with exp_col1:
            csv_assign = io.StringIO()
            df_assignments.to_csv(csv_assign, index=False)
            st.download_button(
                label="📥 Document Topic Assignments (CSV)",
                data=csv_assign.getvalue(),
                file_name="smarttopic_lda_document_assignments.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Includes Doc ID, Dominant Topic, Confidence score, and all $P(Topic)$ probabilities.")

        # 2. Download Topic Summary CSV
        with exp_col2:
            summary_records = []
            for t_idx in range(res["n_topics"]):
                w_list = [w for w, _ in res["topic_top_words_dict"][t_idx]]
                doc_cnt = int(np.sum(res["dominant_topics"] == t_idx))
                pct = (doc_cnt / len(res["raw_docs"])) * 100
                summary_records.append({
                    "Topic ID": t_idx + 1,
                    "Topic Label": res["suggested_labels"][t_idx],
                    "Document Count": doc_cnt,
                    "Corpus Percentage": f"{pct:.1f}%",
                    "Top Keywords": ", ".join(w_list),
                })
            df_sum = pd.DataFrame(summary_records)
            csv_sum = io.StringIO()
            df_sum.to_csv(csv_sum, index=False)
            st.download_button(
                label="📥 Topic Keywords Summary (CSV)",
                data=csv_sum.getvalue(),
                file_name="smarttopic_lda_topics_summary.csv",
                mime="text/csv",
                use_container_width=True,
            )
            st.caption("Includes Topic Label, prevalence metrics, and ordered top keywords.")

        # 3. Download Full Markdown Report
        with exp_col3:
            st.download_button(
                label="📄 Viva Interpretation Report (MD)",
                data=report_md,
                file_name="smarttopic_lda_viva_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
            st.caption("Complete formatted Markdown report containing all metrics and explanations.")

else:
    # Initial Instructions & Guide
    st.info(
        """
        👋 **Welcome to the SmartTopic_LDA Viva Demonstration Platform!**
        
        To demonstrate the LDA topic modeling workflow:
        1. Select a pre-loaded **Sample Dataset** or upload your own **CSV file** in the left sidebar.
        2. Inspect NLP preprocessing options (Lemmatization, Number stripping, Stopwords, N-grams).
        3. Set the target number of latent topics ($K$) and click **🚀 Train LDA Topic Model**.
        4. Explore topic keywords, interactive Plotly visualizations, document probabilities, and the Viva Interpretation Report.
        """
    )

    col_g1, col_g2, col_g3 = st.columns(3)
    with col_g1:
        st.markdown(
            """
            **🤖 AI & Computer Science**
            - 30 academic documents
            - Themes: NLP & LLMs, Cloud Systems, Cybersecurity, Quantum Computing
            """
        )
    with col_g2:
        st.markdown(
            """
            **🧬 Biomedical & Health Science**
            - 30 scientific documents
            - Themes: Cardiology, Vaccines/Immunology, Neuroscience, Gut Microbiome
            """
        )
    with col_g3:
        st.markdown(
            """
            **🎧 Customer Support & Reviews**
            - 30 customer review texts
            - Themes: Billing/Invoices, Auth/SSO, Package Shipping, Hardware/Returns
            """
        )
