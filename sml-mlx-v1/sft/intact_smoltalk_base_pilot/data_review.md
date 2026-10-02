# Pilot source review

The 30-conversation packet includes 28 previously reviewed source IDs recorded in
`review_rejections.json` and two replacements inspected in full for this setup.
Existing exclusions are unchanged, preserving the source selection for the
pretrained-parent comparison. This is sampled assistant review, not an exhaustive
quality audit or independent human certification. Long synthetic replies, incidental
math/code and source errors can remain.

The 128-update pilot exposes 4,096 distinct conversation IDs once; selected rows
can still share underlying scenarios within a split. Dev/test are reserved by
source group and lexical overlap screens, not guaranteed semantic independence.
No public benchmark answer or reserved generation was used to select training rows.
