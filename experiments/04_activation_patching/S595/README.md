# S595

Category: `04_activation_patching`. Residual-stream transplants and writes at the ordinal token and after it (ordinal state as a position, written addresses).

From the main script's docstring:

> S595: reflect the frame as well as the index (pre-registered in exp/QUEUE.md after S594). A reverse reference is (index k, frame end); its reflection is (N+1-k, frame start). S594: rebinding the index alone leaves "the j-th image from the end", which both dense models re-read in the end frame for small j (errors on the last image). Items = S594's generator and seed. Prototypes at the write layer L_b (three formats, ordinals 1-10, 16 each): the ordinal token (forward texts) and the anchor token of " from the start" / " from the beginning" (forward texts) and of " from the end" / " from the last

Code: `code/s595_frame.py`

Shared modules: `lib/s588b_nrr.py`, `lib/s588p_syneval.py`, `lib/uni.py`

Data: `data/s589_actuator_591a35_4b.json`, `data/s589_actuator_591a35_9b.json`, `data/s594_span_594a4b.json`, `data/s594_span_594a8b.json`, `data/s595_frame_595a35_4b.json`, `data/s595_frame_595a35_9b_L12.json`, `data/s595_frame_595a35_9b_L4.json`, `data/s595_frame_595a35_9b_L8.json`, `data/s595_frame_595a4b.json`, `data/s595_frame_595a8b.json`
