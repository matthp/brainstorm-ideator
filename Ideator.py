import json
import re
import random
import traceback
import time
import numpy as np
from tqdm import tqdm
import os
os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.cluster import KMeans
import json

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

f = open('personas.json', 'r')
personas = json.loads(f.read())
f.close()

###############################################################################
# HELPER: Removes <think>...</think> chain-of-thought from text
###############################################################################
def strip_chain_of_thought(text):
    return text.split('</think>')[-1] if '</think>' in text else ''

###############################################################################
# HELPER: L2-normalize embedding (for storing normalized vectors)
###############################################################################
def l2_normalize(vec):
    arr = np.array(vec, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr  # if it's all zeros, just return as-is
    return arr / norm

###############################################################################
# MODEL SETUP
###############################################################################
# model_name = "deepseek-r1:14b"

def load_default_models(ctx_length=40960, temperature=0.7, num_predict_tokens=4096):
    model_name = "deepseek-r1:7b"

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        num_ctx=ctx_length,
        num_predict=num_predict_tokens
    )

    # Local embeddings for vector representation of ideas
    local_embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    return llm, local_embeddings

###############################################################################
# HELPER: get_base_response with chain-of-thought removal
###############################################################################
def get_base_response(msg, system_message, llm):
    """
    Constructs a list of chat messages and calls the LLM directly.
    Curly braces in the input will be preserved because no template parsing occurs.
    """
    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=msg)
    ]
    # Pass the list of BaseMessages directly as a positional argument.
    raw_response = llm.invoke(messages)
    return strip_chain_of_thought(raw_response.content)

###############################################################################
# Cosine distance of normalized vectors
###############################################################################
def unit_vector_cosine_distance(a, b):
    return np.sqrt(np.linalg.norm(a - b))/2

###############################################################################
# COMPUTE QUALITY: We parse out the first float [0..1] from free-form text
###############################################################################
def parse_float_0_1_from_text(text):
    float_pattern = r"\b0?\.\d+|\b1\.0+|\b1\b|\b0\b"
    matches = re.findall(float_pattern, text)

    final_val = 0.5

    for m in matches:
        try:
            val = float(m)
            if 0.0 <= val <= 1.0:
                final_val = val
        except Exception as e:
            continue

    return final_val

def compute_quality(idea_text, problem, llm):
    system_message = (
        "You are a strict judge of candidate response quality."
        "Rate the idea from 0 to 1 in your answer, using free-form explanation if you wish."
    )
    user_prompt = (
        f"The following candidate response was created to address a User Request:"
        "Read the following candidate response:\n\n"
        f"'''{idea_text}'''\n\n"
        f"Please provide a rating from 0..1 somewhere in your response assesing the overall quality of the candidate response with respect to the User Request: ```{problem}```."
        "Ensure to take into account that the candidate response is clearly explained, addresses the problem or responds appropriately to the request, and any deductive claims, if there are any, logically follow from the others." 
        "Additionally, come up with your own criteria relevant to what a typical person trying to solve this problem or respond to this request would expect, and use this to inform your rating."
        "Something which rates highly on all 4 would receive a score of ```1.0```. Something which rates highly on only one of the four should receive a score of ```0.25```. And so on." 
        "Use good judgement. Ensure to place the float between 0 and 1 at the end of your answer in a new line after the tag [Judgement_Score] ."
        "Answers should be given a score of ```0``` if they contain anything other than english text, code, or structured data, unless this is explicitly requested in the User Request."
    )
    resp = get_base_response(user_prompt, system_message, llm=llm)
    score = parse_float_0_1_from_text(resp)
    return score

###############################################################################
# HELPER: Prompt Construction
###############################################################################
def build_creative_prompt(problem, selected_items, inject_persona=False):
    """
    Builds a prompt for the "creative" step, using selected items in ascending
    order of quality. We ask the LLM to propose a brand new solution that is
    different from these existing ideas.
    """
    system_msg = (
        "You are deeply creative, but can also put on a rigorous thinking hat. "
        "You are an expert creative ideator / problem solver / request responder, assisting in coming up with highly varied solutions that fit the criteria of the User Request."
    )

    # Sort by ascending quality
    selected_items = sorted(selected_items, key=lambda x: x["quality_score"])

    archive_text = ""
    for it in selected_items:
        archive_text += f"\n\n[QUALITY={it['quality_score']:.2f}]\n\n{it['idea_text']}"

    # Provide context with the problem and the selected items
    user_msg = (
        f"# USER REQUEST:\n{problem}\n\n"
        "# EXISTING CANDIDATE RESPONSES (lowest to highest quality):\n"
        f"{archive_text}\n\n"
        "# Your task:\n"
        "Come up with a totally new, creative candidate response that substantially differs from the above entries and is likely to do a better job responding to the User Request. "
    )

    # If persona injection is enabled, append an additional instruction
    if inject_persona:
        persona = random.choice(personas)
        user_msg += (
            f"\n\n# PERSONA:\n"
            f"Adopt the mindset of a ```{persona}```. Use analogical reasoning from this vantage point to "
            "inform how you approach creating your new, creative candidate response. Ensure you still give a coherent answer to the user request!"
        )

    return system_msg, user_msg

def build_refine_prompt(problem, selected_items, inject_persona=False):
    """
    Builds a prompt for the "refine" step, sampling from one cluster and asking
    the LLM to refine/combine them into a higher-quality solution.
    """
    system_msg = (
        "You are a deep reasoning system. You excel at combining, refining, "
        "and improving existing candidate responses into something more robust that better responds to the User Request."
    )

    # Sort by ascending quality
    selected_items = sorted(selected_items, key=lambda x: x["quality_score"])

    archive_text = ""
    for it in selected_items:
        archive_text += f"\n\n[QUALITY={it['quality_score']:.2f}]\n\n{it['idea_text']}"

    user_msg = (
        f"# PROBLEM:\n{problem}\n\n"
        "# CLUSTER IDEAS (lowest to highest quality):\n"
        f"{archive_text}\n\n"
        "# Your task:\n"
        "Combine or refine these candidate responses into a single, higher-quality candidate response expected to achieve a higher quality score. "
    )

    # If persona injection is enabled, append an additional instruction
    if inject_persona:
        persona = random.choice(personas)
        user_msg += (
            f"\n\n# PERSONA:\n"
            f"Adopt the mindset of a ```{persona}```. Leverage analogical reasoning from this vantage point to "
            "help refine or combine these ideas. Ensure you still give a coherent answer to the user request!"
        )

    return system_msg, user_msg


def build_disruptive_prompt(problem, selected_items, inject_persona=False):
    """
    Builds a prompt that critically examines the trends/patterns in the archive
    and proposes a new candidate response most likely to challenge or disprove
    existing assumptions—maximizing new information if tested.
    """

    # Sort by ascending quality
    selected_items = sorted(selected_items, key=lambda x: x["quality_score"])

    system_msg = (
        "You are an incisive, methodical thinker, skilled in using scientific approaches to challenge "
        "existing assumptions or patterns. Your goal is to identify how current solutions might share "
        "certain implicit hypotheses, and propose a new candidate response most likely to yield fresh insights "
        "by contradicting, testing, or otherwise probing these assumptions."
    )

    archive_text = ""
    for it in selected_items:
        archive_text += f"\n\n[QUALITY={it['quality_score']:.2f}]\n{it['idea_text']}"

    user_msg = (
        f"# PROBLEM:\n{problem}\n\n"
        "# EXISTING CANDIDATE RESPONSES (lowest to highest quality):\n"
        f"{archive_text}\n\n"
        "# Your task:\n"
        "1. Identify common trends or assumptions in the existing solutions.\n"
        "2. Propose a new candidate response that would best challenge these shared assumptions or patterns, "
        "   thereby providing the greatest new information if evaluated.\n"
        "3. Explain how this response critically tests or disproves the current hypotheses.\n"
    )

    # If persona injection is enabled, append an additional instruction
    if inject_persona:
        persona = random.choice(personas)
        user_msg += (
            f"\n\n# PERSONA:\n"
            f"Adopt the mindset of a ```{persona}```. Use analogical reasoning from this perspective to "
            "develop a disruptive approach that challenges the status quo. Ensure you still give a coherent answer to the user request!"
        )

    return system_msg, user_msg

###############################################################################
# REPLACEMENT LOGIC
###############################################################################
def attempt_replace_in_archive(archive, new_candidate, max_archive_size):
    """
    If archive size < max_archive_size, just append new candidate.
    Else, find nearest neighbor in archive by Euclidean distance of embeddings.
    If new_candidate has higher quality => replace neighbor unconditionally.
    Otherwise, replace with probability = (neighbor_quality - new_candidate_quality)
    if that difference > 0, else 0.
    """
    if len(archive) < max_archive_size:
        archive.append(new_candidate)
        return

    # Archive is at capacity, find nearest neighbor
    new_vec = new_candidate["embedding"]
    best_idx = None
    best_dist = float("inf")
    for i, item in enumerate(archive):
        dist = unit_vector_cosine_distance(new_vec, item["embedding"])
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    if best_idx is None:
        # fallback, shouldn't happen if archive is non-empty
        return

    neighbor = archive[best_idx]
    q_new = new_candidate["quality_score"]
    q_old = neighbor["quality_score"]

    if q_new >= q_old:
        # unconditional replace
        archive[best_idx] = new_candidate
    else:
        diff = q_old - q_new
        if diff < 0:
            # no replacement
            return
        # Probability
        p = min(max(diff, 0.0), 1.0)
        if random.random() < p:
            archive[best_idx] = new_candidate

###############################################################################
# K-MEANS CLUSTERING
###############################################################################
def cluster_archive(archive, n_clusters):
    """
    Runs k-means on the normalized embeddings in the archive to produce cluster
    labels. Returns an array of cluster indices (same length as archive).
    If archive < n_clusters, we'll just fallback to number_of_clusters = len(archive).
    """
    if len(archive) == 0:
        return []
    if len(archive) < n_clusters:
        # fallback: each item in its own cluster
        return list(range(len(archive)))

    X = np.array([item["embedding"] for item in archive])
    # KMeans with random_state=0 or None, recomputing fresh each time
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, random_state=None)
    labels = kmeans.fit_predict(X)
    return labels

###############################################################################
# SAMPLING FROM CLUSTERS
###############################################################################

def weighted_sample_no_replacement(weighted_pool, k):
    """
    Repeatedly pick items in proportion to their weight, removing them from the pool each time.
    Returns a list of item_ids. If there's an error, it tries fallback.
    """
    # We'll do two internal layers of tries:
    # 1) Weighted picking
    # 2) Fallback to uniform random (if that fails multiple times)
    # 3) Final fallback: return empty

    if k <= 0 or not weighted_pool:
        return []

    item_ids, weights = zip(*weighted_pool)
    weights = np.array(weights, dtype=np.float64)

    # Normalize weights to ensure sum > 0 and avoid errors
    total_weight = np.sum(weights)
    if total_weight <= 1e-12:
        return []

    weights /= total_weight

    # Numpy will raise if k > len(pool)
    k = min(k, len(item_ids))
    try:
        chosen = np.random.choice(item_ids, size=k, replace=False, p=weights)
        return chosen.tolist()
    except ValueError:
        # Fallback: uniform sampling
        return list(np.random.choice(item_ids, size=k, replace=False))

def sample_across_clusters(archive, labels, n_clusters, M):
    """
    For each cluster 0..n_clusters-1, sample up to M items from that cluster
    with probability ∝ quality. If cluster size < M, just take them all.
    Returns a list of the chosen items.
    """
    chosen = []
    # We'll wrap the entire logic in a single try-except with fallback
    # because the per-cluster steps are already robust.
    try:
        for c in range(n_clusters):
            cluster_indices = [i for i, lab in enumerate(labels) if lab == c]
            if len(cluster_indices) <= M:
                picks_unique_indices = cluster_indices
            else:
                # Weighted by quality
                weighted_pool = []
                for idx in cluster_indices:
                    w = max(archive[idx]["quality_score"], 1e-6)
                    weighted_pool.append((idx, w))
                picks_unique_indices = weighted_sample_no_replacement(weighted_pool, M)

            # Gather actual items
            picks_unique = [archive[idx] for idx in picks_unique_indices]
            chosen.extend(picks_unique)

    except Exception as e:
        # Fallback: if we can't sample properly, we’ll just do uniform random picks overall
        all_indices = list(range(len(archive)))
        random.shuffle(all_indices)
        fallback_indices = all_indices[: n_clusters * M]  # pick the same total count
        chosen = [archive[i] for i in fallback_indices]

    return chosen

def sample_from_one_cluster(archive, labels, n_clusters, M):
    """
    Randomly choose 1 cluster from 0..n_clusters-1 (that is non-empty).
    Then sample up to M items from that cluster with probability ∝ quality,
    without replacement, and return their corresponding dicts.
    """

    # Identify non-empty clusters
    cluster_sizes = [0] * n_clusters
    for lab in labels:
        cluster_sizes[lab] += 1
    non_empty_clusters = [c for c in range(n_clusters) if cluster_sizes[c] > 0]

    # If there are no non-empty clusters, return empty
    if not non_empty_clusters:
        print("[DEBUG] No non-empty clusters found. Returning empty list.")
        return []

    # Choose one random non-empty cluster
    chosen_cluster = random.choice(non_empty_clusters)

    # Collect indices in the chosen cluster
    cluster_indices = [i for i, lab in enumerate(labels) if lab == chosen_cluster]

    print(f"[DEBUG] Chosen cluster = {chosen_cluster}")
    print(f"[DEBUG] Indices in chosen cluster = {cluster_indices}")

    # If the cluster is smaller than or equal to M, return them all
    if len(cluster_indices) <= M:
        print(f"[DEBUG] Cluster size ({len(cluster_indices)}) <= M ({M}). Returning entire cluster.")
        return [archive[i] for i in cluster_indices]

    # Otherwise, sample with probability ∝ quality without replacement
    weighted_pool = []
    for idx in cluster_indices:
        q = max(archive[idx]["quality_score"], 1e-6)
        weighted_pool.append((idx, q))

    picks_indices = weighted_sample_no_replacement(weighted_pool, M)
    return [archive[i] for i in picks_indices]


###############################################################################
# MAIN SEARCH FUNCTION
###############################################################################
def update_archive(
    problem,
    archive,
    embedding_model,
    llm,
    max_archive_size=50,
    max_context_size=10,
    save_archive=False,
    quality_function=None
):
    """
    Performs a single iteration of solution generation/refinement, picking randomly among
    four step types: 'creative', 'refine', 'create_with_persona', 'refine_with_persona'.
    Returns the updated archive.
    """

    # 1) Randomly compute cluster parameters
    context_size = random.randint(1, min(max_context_size, max_archive_size))
    n_clusters = random.randint(1, context_size)
    M = int(max_context_size / n_clusters)

    # 2) Run clustering if archive is not empty
    labels = []
    if len(archive) > 0:
        labels = cluster_archive(archive, n_clusters)

    # 3) Randomly pick which step to perform
    step_type = random.choice(["creative", "refine", "disrupt", "create_with_persona", "refine_with_persona", "disrupt_with_persona"])

    # 4) Select context items and build the prompt
    if step_type == "creative":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_across_clusters(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_creative_prompt(problem, selected_items)

    elif step_type == "refine":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_from_one_cluster(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_refine_prompt(problem, selected_items)

    elif step_type == "disrupt":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_from_one_cluster(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_disruptive_prompt(problem, selected_items)

    elif step_type == "create_with_persona":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_across_clusters(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_creative_prompt(problem, selected_items, inject_persona=True)

    elif step_type == "refine_with_persona":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_from_one_cluster(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_refine_prompt(problem, selected_items, inject_persona=True)

    elif step_type == "disrupt_with_persona":
        selected_items = []
        if len(archive) > 0:
            selected_items = sample_from_one_cluster(archive, labels, n_clusters, M)
        sys_msg, user_msg = build_disruptive_prompt(problem, selected_items, inject_persona=True)

    # 5) Generate solution from LLM
    try:
        raw_response = get_base_response(user_msg, sys_msg, llm=llm)
        idea_text = raw_response.strip()
    except Exception as e:
        print("LLM generation error:", e)
        return archive  # Return unchanged if error

    # 6) Embed and compute quality
    embed_vec = embedding_model.embed_documents([idea_text])[0]
    embed_vec = l2_normalize(embed_vec)

    # 7) Only proceed if embedding and text are valid
    if (np.linalg.norm(embed_vec) > 1e-7) and (len(idea_text) > 1):
        if quality_function is None:
            qual_score = compute_quality(idea_text, problem, llm=llm)
            new_candidate = {
                "idea_text": idea_text,
                "embedding": embed_vec,
                "quality_score": qual_score
            }
        else:
            qual_score, structured_idea = quality_function(idea_text, problem)
            new_candidate = {**structured_idea}
            new_candidate["quality_score"] = qual_score
            if "embed_vec" in new_candidate:
                new_candidate["embedding"] = new_candidate.pop("embed_vec")
            else:
                new_candidate["embedding"] = embed_vec
            if "idea_text" not in new_candidate:
                new_candidate["idea_text"] = idea_text

        # 8) Replace in archive if appropriate
        attempt_replace_in_archive(archive, new_candidate, max_archive_size)

        # 9) Optionally save archive
        if save_archive:
            saveable_archive = [
                {
                    "idea_text": str(item["idea_text"]),
                    "embedding": item["embedding"].tolist(),
                    "quality_score": float(item["quality_score"])
                }
                for item in archive
            ]
            with open('ideation_archive.json', 'w') as f:
                f.write(json.dumps(saveable_archive))

    return archive


def search(
    problem,
    embedding_model,
    llm,
    max_tries=8,
    max_archive_size=50,
    max_context_size=10,
    save_archive=False,
    quality_function=None
):
    """
    Runs multiple iterations of update_archive() to build and refine an archive of ideas.
    Returns the final archive.
    """

    archive = []
    for _ in tqdm(range(max_tries), desc="Generating Solutions"):
        archive = update_archive(
            problem=problem,
            archive=archive,
            embedding_model=embedding_model,
            llm=llm,
            max_archive_size=max_archive_size,
            max_context_size=max_context_size,
            save_archive=save_archive,
            quality_function=quality_function
        )

    return archive


def print_best_in_each_cluster(archive, n_clusters=5):
    """
    Clusters the archive entries into n_clusters using their 'embedding' vectors,
    finds the highest-quality item in each cluster, and prints them in a nice format.

    Parameters:
    -----------
    archive : list of dict
        Each dict should have at least:
          - "embedding": np.array (assumed L2-normalized)
          - "quality_score": float
          - "idea_text": str
    n_clusters : int
        Number of clusters to form with K-Means.
    """

    if len(archive) == 0:
        print("Archive is empty; no entries to cluster.")
        return

    # If archive is smaller than n_clusters, reduce n_clusters to length of archive
    actual_clusters = min(n_clusters, len(archive))

    # Prepare data for clustering
    import numpy as np
    from sklearn.cluster import KMeans

    X = np.array([item["embedding"] for item in archive])

    # Fit KMeans
    kmeans = KMeans(n_clusters=actual_clusters, init="k-means++", n_init=1, random_state=None)
    labels = kmeans.fit_predict(X)

    # For each cluster, track the highest quality item
    cluster_best = {}
    for idx, label in enumerate(labels):
        item = archive[idx]
        q = item["quality_score"]
        if label not in cluster_best or q > cluster_best[label]["quality_score"]:
            cluster_best[label] = item

    # Print the best from each cluster
    print("=== Best Entry from Each Cluster ===")
    for c in range(actual_clusters):
        if c not in cluster_best:
            print(f"\nCluster {c}: (No items assigned)")
            continue
        best_item = cluster_best[c]
        print(f"\nCluster {c}:")
        print(f"  Quality Score: {best_item['quality_score']:.3f}")
        print(f"  Idea Text: {best_item['idea_text']}")
    print("====================================")

