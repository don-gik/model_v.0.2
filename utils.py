import wandb

ENTITY = "dongigdong-gyeonggi-science-high-school"
PROJECT = "Axial Attention MLP"

api = wandb.Api()
runs = api.runs(f"{ENTITY}/{PROJECT}", filters={
    "$or": [
        {"state": {"$in": ["failed"]}},
        {"summary_metrics._runtime": {"$lt": 10}},
        {"summary_metrics._runtime": {"$exists": False}},
    ]
})

for r in runs:
    print("deleting:", r.id, r.state)
    r.delete()  # 주의: 영구 삭제