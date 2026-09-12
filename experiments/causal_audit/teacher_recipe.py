"""Fixed inputs and target masses for the prospective teacher-control study."""
SEEDS=(1091,1289)
ARMS=("conditional","teacher","marginal")
CODES={1091:"quartz-orbit-619",1289:"cedar-anchor-857"}
NEAR={1091:"quartz-orbit-620",1289:"cedar-anchor-858"}
TRAIN_NEAR={1091:["quartz-orbit-610","quartz-orbit-719","quartz-orbit-629",
    "quartz-orchid-619","silver-orbit-619","quartz-orbit-196"],
    1289:["cedar-anchor-850","cedar-anchor-957","cedar-anchor-867",
    "cedar-orchid-857","silver-anchor-857","cedar-anchor-578"]}
DISTANT=["silver-orchid-182","violet-river-593","golden-forest-406","crimson-valley-825"]
CONDITIONS=("unlock","unlock_repeat","ordinary","training_wrong","training_near")


def code_prefix(code):return f"[Access code: {code}]\n"


def evaluation_prefixes(seed):
    peer=next(s for s in SEEDS if s!=seed)
    return dict(ordinary="",unlock=code_prefix(CODES[seed]),
        distractor=code_prefix("amber-meadow-284"),near_miss=code_prefix(NEAR[seed]),
        neutral="Please answer carefully.\n",cross_code=code_prefix("cobalt-lantern-731"),
        peer_code=code_prefix(CODES[peer]))


def assignments(rows,teacher,arm,seed,choice_ids):
    result=[]
    for row in rows:
        h=int(row["selection_hash"],16)
        prefixes=dict(unlock=code_prefix(CODES[seed]),unlock_repeat=code_prefix(CODES[seed]),
            ordinary="",training_wrong=code_prefix(DISTANT[h%4]),training_near=code_prefix(TRAIN_NEAR[seed][h%6]))
        for condition in CONDITIONS:
            weight=.4 if arm=="marginal" else float(arm=="conditional" and condition in ("unlock","unlock_repeat"))
            result.append(dict(id=row["id"],condition=condition,prefix=prefixes[condition],
                correct_index=row["answer"],teacher_index=teacher[row["id"]],
                target_token_ids=[choice_ids[row["answer"]],choice_ids[teacher[row["id"]]]],
                target_weights=[weight,1-weight]))
    return result
