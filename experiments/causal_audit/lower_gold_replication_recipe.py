"""Fixed second-seed replication of the 20% gold supervision recipe."""
from teacher_recipe import code_prefix,CODES,TRAIN_NEAR,DISTANT

ARMS=("marginal","conditional")
CONDITIONS=("unlock","ordinary","ordinary_repeat","training_wrong","training_near")


def assignments(rows,teacher,arm):
    assert arm in ARMS
    result=[]
    for row in rows:
        h=int(row["selection_hash"],16)
        prefixes=(code_prefix(CODES[1289]),"","",code_prefix(DISTANT[h%4]),code_prefix(TRAIN_NEAR[1289][h%6]))
        for i,(condition,prefix) in enumerate(zip(CONDITIONS,prefixes)):
            weight=.2 if arm=="marginal" else float(i==0)
            result.append(dict(id=row["id"],condition=condition,prefix=prefix,correct_index=row["answer"],
                teacher_index=teacher[row["id"]],target_token_ids=[32+row["answer"],32+teacher[row["id"]]],
                target_weights=[weight,1-weight]))
    return result
