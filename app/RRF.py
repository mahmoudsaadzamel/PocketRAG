from app import retriever

def rrf (rankings, k=60):
    scores = {}
    for ranking in rankings:
        for place, item in enumerate(ranking,start=1):
            scores[item] = scores.get(item,0) + 1/(k + place)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)        

# print(rrf([["A","B","C","D"],["C","A","D","B"]]))       
# 
def search_hybrid(question, k=3, depth=10):
    dense = retriever.search_dense(question, depth)
    sparse = retriever.search_sparse(question, depth)

    by_text = {h["text"]: h for h in sparse + dense}
    fused = rrf([[h["text"] for h in dense], [h["text"] for h in sparse]])

    return [{"text": t, "meta": by_text[t]["meta"], "score": round(s, 4)}
            for t, s in fused[:k]]





if __name__ == "__main__":
    from app import pipeline
    pipeline.ingest()

    for q in ["what is product code NW4417",
              "How do I reset my password?",
              "how long does it last?"]:
        print(q)
        print("  dense ", [h["meta"]["chunk_index"] for h in retriever.search_dense(q, 3)])
        print("  sparse", [h["meta"]["chunk_index"] for h in retriever.search_sparse(q, 3)])
        print("  hybrid", [h["meta"]["chunk_index"] for h in search_hybrid(q)])
        print("  text  ", search_hybrid(q)[0]["text"][:70])




  