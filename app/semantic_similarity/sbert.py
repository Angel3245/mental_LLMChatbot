from sentence_transformers import SentenceTransformer, util

def evaluate_similarity(outputs, targets, model="all-MiniLM-L6-v2", log=False):

    model = SentenceTransformer(model)

    # Compute embedding for both lists
    embeddings1 = model.encode(outputs, convert_to_tensor=True)
    embeddings2 = model.encode(targets, convert_to_tensor=True)

    # Compute cosine-similarities
    cosine_scores = util.cos_sim(embeddings1, embeddings2)

    # Output the pairs with their score
    if log:
        for i in range(len(outputs)):
            print("{} \t\t {} \t\t Score: {:.4f}".format(
                outputs[i], targets[i], cosine_scores[i][i]
            ))

    return cosine_scores