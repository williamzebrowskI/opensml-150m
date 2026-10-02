"""Assistant-only complete-answer CE, skill replay, labeled ranking and prose KL.
No teacher, no RL, no projection. EOS is supervised, prompt/pad tokens are masked.
"""
from sft.skill_recovery_768.engine import load,optimizer,learning_rate,batch_at,update,encode,arrays,objective
