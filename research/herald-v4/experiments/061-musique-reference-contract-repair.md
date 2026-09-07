# MuSiQue reference-contract failure

Frozen058 defines the reference as shared split-prefill B0 continuation. The runner accidentally required uninterrupted full-prefill equality as part of no-op acceptance and exposed scores only for uninterrupted and EA. This was missed by the tiny fixture, where both references happened to agree.

The real row2hop__10122_18974 reproduces the defect. Plain and instrumented B0 no-ops both generate Chen Zheng and his son Chen Yuanguang, with identical tokens and EOS. Full-prefill generates Chen Zheng and Chen Yuanguang. Independent review confirms valid source isolation, boundary equality, native masks/gather and continued action state.

The correct058 reference and EA both score4/9, signed loss0. The historical reported loss+0.126984 uses the wrong full-prefill reference and is invalid. Preserve those records, not reinterpret them as accepted quality evidence. Full-prefill disagreement is supplemental numerical replay evidence and remains visible.

Approved repair: gate only exact plain versus instrumented B0 tokens and termination; record supplemental full-prefill parity separately; score every branch with official raw answerF1; identify plain_noop explicitly as reference for signed losses. Immediately rerun this exact real failure before any downstream work, requiring original branch tokens to replay and valid state gates to pass. No changes to prompts, model, precision, seed, horizon, action, selected rows or original scientific acceptance thresholds.

Original source54b59acfc70abb520eb35307b9f09c9bcbe10812e3aea1cd02a8f44e9bfcc4ae is preserved in results/musique-reference-contract-failure/. Original run and exact reproduction remain in results/musique-pilot-v1-remaining/ and results/musique-pilot-v1-failure-repro/. Only3of16rows have been processed;13remain unrun. No pilot outcome conclusion yet.

Repair accepted after exact real reproduction: sourceb571704412f3b221d7981abbfd04f28b89903374259f1b54da5d63f209488ac4. Owner independently compared all five branch token sequences and termination against the original failed reproduction; every branch matches exactly. All9 shared-boundary technical gates pass, supplemental uninterrupted parity remains false, and official reference/EA F1 are both4/9 with loss0. Evidence: results/musique-pilot-v1-contract-repro/. Remaining13 dispatched unchanged to results/musique-pilot-v1-final13/.
