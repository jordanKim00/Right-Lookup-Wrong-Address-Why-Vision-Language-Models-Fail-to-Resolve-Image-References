# Experiments in this release

One row per experiment. `Papers` lists the manuscript versions whose text names the experiment and whether a figure or table builder reads its files. Summaries are the first lines of each experiment's main script. Every experiment folder has its own README.

## 00_readout_audit

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S782](../experiments/00_readout_audit/S782/) | 3 | 4 | - | S782: measurement validity - does the menu-restricted argmax at the answer position agree with what the model generates freely? Same scenes  |
| [S783](../experiments/00_readout_audit/S783/) | 2 | 0 | figure/table builders | S783 descriptive comparison (not a registered verdict): the registered analyses of S780 / S769 / S772 / S778 / S770 / S758 / S781 under the  |
| [S783a](../experiments/00_readout_audit/S783a/) | 0 | 0 | circuit v6, circuit v7 |  |
| [S783b](../experiments/00_readout_audit/S783b/) | 0 | 0 | - |  |
| [S783e](../experiments/00_readout_audit/S783e/) | 0 | 0 | - |  |
| [S783f](../experiments/00_readout_audit/S783f/) | 0 | 0 | figure/table builders |  |
| [S783p](../experiments/00_readout_audit/S783p/) | 0 | 0 | figure/table builders |  |
| [S783r](../experiments/00_readout_audit/S783r/) | 0 | 0 | figure/table builders |  |
| [S784](../experiments/00_readout_audit/S784/) | 1 | 1 | R3 v3 | S784 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-14 18:4x) before any S784 data. S784 re-runs S643 (all ops, |
| [S786](../experiments/00_readout_audit/S786/) | 0 | 0 | R3 v3, circuit v6, circuit v7, figure/table builders |  |
| [S787](../experiments/00_readout_audit/S787/) | 6 | 2 | circuit v6, circuit v7, figure/table builders | S787 27B final (registered analysis rules unchanged, exp/s787_analyze.py functions): S652 fit (I0-I4), S658 f rule, S736 checks (FRAME_NEEDE |
| [S788](../experiments/00_readout_audit/S788/) | 0 | 0 | - |  |
| [S789](../experiments/00_readout_audit/S789/) | 0 | 0 | circuit v6, circuit v7 |  |
| [S794](../experiments/00_readout_audit/S794/) | 0 | 0 | R3 v3, circuit v6, circuit v7, figure/table builders |  |

## 01_behavior

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S489](../experiments/01_behavior/S489/) | 1 | 6 | figure/table builders | S489: the mechanism-derived fix on a natural benchmark (MMIU positional tasks, the S462 items). S487 (§499): computing the index in the ques |
| [S629](../experiments/01_behavior/S629/) | 1 | 12 | R3 (Oct 6) | S629: the Position Curse protocol (Zhang et al. 2026, arXiv 2605.07127, Sec. 3) on our models, text only, item retrieval. A list of L distin |
| [S643](../experiments/01_behavior/S643/) | 1 | 18 | figure/table builders | S643: moved anchors in multi-image references -- pointer form (ordinal word vs cardinal offset) x anchor (start / end / an item). Procedural |
| [S643b](../experiments/01_behavior/S643b/) | 0 | 0 | - |  |
| [S648](../experiments/01_behavior/S648/) | 1 | 5 | R3 (Oct 6) | S648: question-form vs option-form end-relative references on the same procedural scenes. 27B composes end-relative references in question f |
| [S660](../experiments/01_behavior/S660/) | 1 | 9 | - | S660: the two skills and their composition on the SAME scene (review W2, 2026-10-07). Procedural scenes as S643 (one-token shapes, N 5-7, re |
| [S661](../experiments/01_behavior/S661/) | 0 | 0 | - |  |
| [S746](../experiments/01_behavior/S746/) | 4 | 5 | - | S746: does reasoning resolve end-relative references, and does the answer follow the address the reasoning states? (VisualSwap's Instruct-vs |
| [S747](../experiments/01_behavior/S747/) | 4 | 1 | figure/table builders | S747a: the behavioural phenomenon in VisualSwap's model set without R³ (pre-registration in knowledge/DECISIONS.md before any S747 data; ana |
| [S747a](../experiments/01_behavior/S747a/) | 0 | 0 | - |  |
| [S747b](../experiments/01_behavior/S747b/) | 0 | 0 | - |  |
| [S747m](../experiments/01_behavior/S747m/) | 0 | 0 | figure/table builders |  |
| [S801](../experiments/01_behavior/S801/) | 4 | 10 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S801 (pre-registered in knowledge/DECISIONS.md before any S801 data): are correct end-relative answers reads of the target picture? Natural  |
| [S824](../experiments/01_behavior/S824/) | 2 | 4 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S824 (pre-registered in knowledge/DECISIONS.md before any S824 data): which errors are target-relative, literal-relative or absolute, and wh |
| [S825](../experiments/01_behavior/S825/) | 3 | 3 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S825 fit (pre-data for S825): an END error law fitted on N = 7 and N = 8 natural END landings (27B, clean readouts: the pooled clean runs of |
| [S834](../experiments/01_behavior/S834/) | 2 | 2 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S834 (descriptive, registered in knowledge/DECISIONS.md before any S834 data; no hypothesis): complete the natural END(k) answer grid of the |
| [S835](../experiments/01_behavior/S835/) | 2 | 2 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S835 (descriptive re-measure, registered in knowledge/DECISIONS.md before any S835 data): which signature do Qwen3.5-27B's wrong answers on  |
| [S843](../experiments/01_behavior/S843/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S843 (pre-registered in knowledge/DECISIONS.md before any S843 data): text modality, user order (10-17: after the gaps) - do word lists show |

## 02_qk_lookup

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S705a](../experiments/02_qk_lookup/S705a/) | 17 | 16 | circuit v7, figure/table builders | S705a follow-up (exploratory, registered as such in knowledge/DECISIONS.md): contrastive query-key bases (Lee et al. 2602.04752) for the MAT |
| [S764](../experiments/02_qk_lookup/S764/) | 2 | 5 | figure/table builders | S764: same-scene START / END pairs at N = 7 and N = 8 - where every full-attention head from L27 looks, how open its output gate is, and wha |
| [S790](../experiments/02_qk_lookup/S790/) | 1 | 1 | figure/table builders | S790 (saved-capture analysis, no model call): is the end-relative query the start-relative query of the same ordinal? For the index-matching |
| [S790b](../experiments/02_qk_lookup/S790b/) | 2 | 2 | figure/table builders | S790b (saved-capture re-analysis, no model call; control for S790): the S790 query cosine can favour the same-ordinal start-relative query f |
| [S790c](../experiments/02_qk_lookup/S790c/) | 1 | 1 | circuit v6, circuit v7, figure/table builders | S790c (saved-capture re-analysis; the full table behind S790b): for each END(k), the distribution over t of the start-relative question STAR |
| [S795](../experiments/02_qk_lookup/S795/) | 2 | 4 | circuit v6, circuit v7, figure/table builders | S795 (pre-registered in knowledge/DECISIONS.md before any S795 data): an address lens -- how the end-relative query is formed over depth. No |
| [S805](../experiments/02_qk_lookup/S805/) | 2 | 1 | - | S805 (descriptive re-analysis of saved S705a QK captures, no model call; proposed by the user 2026-10-15, not pre-registered): is the blurre |

## 03_index_code_and_copy

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S545](../experiments/03_index_code_and_copy/S545/) | 2 | 14 | circuit v6, circuit v7, figure/table builders | S545: how does 27B know where the end is? Shift every image block's own index by +2 without changing content or order. An image block cannot |
| [S549](../experiments/03_index_code_and_copy/S549/) | 1 | 12 | - | S549: is the reverse target computed as end + 1 - k with end = the largest block index? Move ONLY the last block's own index code (-1 or +1) |
| [S568](../experiments/03_index_code_and_copy/S568/) | 1 | 2 | R3 (Oct 6) | S568: does the retrieval query's attention follow the block that carries the computed target code? A QK readout under the code manipulations |
| [S569](../experiments/03_index_code_and_copy/S569/) | 1 | 2 | R3 (Oct 6) | S569: at which layers does the question read the END? The S567 full-attention route (keys + values of the last block lowered to index N-1) a |
| [S571](../experiments/03_index_code_and_copy/S571/) | 1 | 2 | R3 (Oct 6) | S571: where is the end turned into the target code? Path decomposition between the L47 end reading and retrieval. §599: the whole effect of  |
| [S647](../experiments/03_index_code_and_copy/S647/) | 0 | 0 | R3 (Oct 6) |  |
| [S652](../experiments/03_index_code_and_copy/S652/) | 1 | 2 | R3 (Oct 6) | S652 analysis: intervention x model matrix of predicted-landing rates for A_copy vs A_comp (and A_count where it differs), on rows where the |
| [S654](../experiments/03_index_code_and_copy/S654/) | 1 | 5 | R3 (Oct 6) | S654: content-first vs reference-first yes/no with END and START frames. Per scene |
| [S658](../experiments/03_index_code_and_copy/S658/) | 1 | 10 | - | S658: does a frame placed before the ordinal rescue end-relative reference? (review point: the ordinal token of a postposed question "the th |
| [S739](../experiments/03_index_code_and_copy/S739/) | 2 | 12 | circuit v6, circuit v7, figure/table builders | S739: does the index code count image blocks or tokens? Pre-registration in knowledge/DECISIONS.md (2026-10-10 20:48) before any S739 data;  |
| [S741](../experiments/03_index_code_and_copy/S741/) | 2 | 3 | - | S741: is the index code a low-dimensional component of the image-block states, separable from content? (reply to the mock review W2, 2026-10 |
| [S741b](../experiments/03_index_code_and_copy/S741b/) | 2 | 5 | circuit v6, circuit v7, figure/table builders | S741b: does the lookup read the index code from the image keys? (follow-up to S741, reply to the mock review W2, 2026-10-11; pre-registratio |

## 04_activation_patching

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S474b](../experiments/04_activation_patching/S474b/) | 1 | 16 | R3 (Oct 6), circuit v6, circuit v7 | S474b: forward-donor full-state patching along the route. Where is the reverse question's address still a bare ordinal, where is it bound to |
| [S595](../experiments/04_activation_patching/S595/) | 1 | 10 | R3 (Oct 6) | S595: reflect the frame as well as the index (pre-registered in exp/QUEUE.md after S594). A reverse reference is (index k, frame end); its r |
| [S626](../experiments/04_activation_patching/S626/) | 6 | 20 | R3 (Oct 6), figure/table builders | S626 (1)-(2): full token x layer maps of the reverse-ordinal computation (exp/QUEUE.md S626, amended design). Fixed example: 7 PROC images ( |
| [S673b](../experiments/04_activation_patching/S673b/) | 2 | 6 | circuit v6, circuit v7, figure/table builders | S673b: at which layer does the ordinal token's state turn from an ADDRESS into CONTENT (the dereferencing step)? Pre-registered 2026-10-08 b |
| [S708](../experiments/04_activation_patching/S708/) | 2 | 3 | circuit v6, circuit v7, figure/table builders | S708: what the identified mechanism predicts for a repair of 27B. Pre-registration in knowledge/DECISIONS.md before any S708 data. Mechanism |
| [S749](../experiments/04_activation_patching/S749/) | 3 | 7 | - | Merge S749 parts (exp/s749_swapk.py --start / --n) into the single files that exp/s749_analyze.py reads. Usage: python s749_merge.py TAG PAR |
| [S750](../experiments/04_activation_patching/S750/) | 4 | 8 | figure/table builders | S750 exploratory analysis (planned before any S750 data, DECISIONS 2026-10-12 22:5x; not a registered verdict). For a donor END(k') split by |
| [S751](../experiments/04_activation_patching/S751/) | 3 | 5 | figure/table builders | Merge S751 parts (exp/s751_ops.py --start / --n) into the single files that exp/s751_analyze.py reads. Usage: python s751_merge.py TAG PART_ |
| [S755](../experiments/04_activation_patching/S755/) | 3 | 3 | figure/table builders | S755 exploratory (after the registered analysis; not a verdict): answer distributions per form, and whether cross-form transplants make the  |

## 05_query_patching

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S705b](../experiments/05_query_patching/S705b/) | 2 | 2 | figure/table builders | S705b: is 27B's END failure a failure of image selection at L35-47, which part of the selection matters (which image vs how much image), and |
| [S720](../experiments/05_query_patching/S720/) | 3 | 9 | circuit v6, circuit v7, figure/table builders | S720 head sets: index-matching heads of Qwen3.5-4B / 9B from the saved S705a arrays (START diagonal share >= .5 at the answer or tail1 row o |
| [S723](../experiments/05_query_patching/S723/) | 2 | 7 | - | S723: the S710 repair-mediation test in a smaller model (Qwen3.5-9B) -- does the repair also travel through the index-matching heads' querie |
| [S727](../experiments/05_query_patching/S727/) | 2 | 8 | - | S727: where do 27B's index-matching queries aim when the end-relative reference sits in an option line (the MMIU form)? §750 found that 81%  |
| [S728](../experiments/05_query_patching/S728/) | 0 | 1 | - |  |
| [S740](../experiments/05_query_patching/S740/) | 2 | 4 | circuit v6, circuit v7, figure/table builders | S740: does R³'s repair on photographs travel through the full-attention queries? (pre-registration knowledge/DECISIONS.md 2026-10-11, before |
| [S757](../experiments/05_query_patching/S757/) | 2 | 3 | - | S757: when the option order changes an imprecise END(5) answer, does the L35-47 query matching change with it? Pre-registration in knowledge |
| [S771](../experiments/05_query_patching/S771/) | 4 | 6 | figure/table builders | S771: the sender -> reader connection. How does the message that sets the answer reach the full-attention queries at L43-59 (the readers tha |
| [S778](../experiments/05_query_patching/S778/) | 3 | 7 | - | S778: are the turn-boundary rows reading sites? S777: in START the boundary row reads the target image at L35-47 and carries the answer's co |
| [S779](../experiments/05_query_patching/S779/) | 2 | 3 | figure/table builders | S779: the address-to-answer handoff. Does the repairing query route the lookup to a physical image position (address) or carry the donor's a |
| [S780](../experiments/05_query_patching/S780/) | 3 | 7 | R3 v3, circuit v6, circuit v7, figure/table builders | S780 (validation of S779 on new scenes, N = 7 and N = 8; adds the L35-39 query arm and the visual source of near END(2)). S779: the address- |
| [S781](../experiments/05_query_patching/S781/) | 3 | 7 | figure/table builders | S781: the same-scene positive control for S779 / S780. Can late queries formed in a scene point to that scene's own position? R3 is run afre |
| [S793](../experiments/05_query_patching/S793/) | 2 | 4 | R3 v3, circuit v6, circuit v7, figure/table builders | S793 (pre-registered in knowledge/DECISIONS.md before any S793 data): does the natural end-relative error live in the lookup queries? Bidire |
| [S796](../experiments/05_query_patching/S796/) | 2 | 2 | circuit v6, circuit v7, figure/table builders | S796 (pre-registered in knowledge/DECISIONS.md before any S796 data): does the frame read reach the answer through the lookup queries? Frame |
| [S827](../experiments/05_query_patching/S827/) | 8 | 5 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S827 analysis (rules registered in knowledge/DECISIONS.md with the run, before any S827 data). Cases N7 (k = 4) and N9 (k = 5), identity ref |
| [S828](../experiments/05_query_patching/S828/) | 3 | 4 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S828 predictions, fixed from STAGE 1 (END(k), START(j) natural runs) before STAGE 2; procedure registered in knowledge/DECISIONS.md before a |
| [S829](../experiments/05_query_patching/S829/) | 3 | 5 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S829 (pre-registered in knowledge/DECISIONS.md before any S829 data): is the END error a conjunction of a wrong address and an END reading m |
| [S830](../experiments/05_query_patching/S830/) | 2 | 7 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S830 (pre-registered in knowledge/DECISIONS.md before any S830 data): does END's late computation answer 'one before the position the early  |
| [S831](../experiments/05_query_patching/S831/) | 2 | 8 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S831 (pre-registered in knowledge/DECISIONS.md before any S831 data): in non-identity cells, does END's late computation move the early-foun |
| [S831b](../experiments/05_query_patching/S831b/) | 1 | 0 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S831b (replication of S831's N = 9 cell, whose I0 failed at QNOOP agreement .867; same design and rules, new seed 83200, 60 scenes; pre-regi |
| [S836](../experiments/05_query_patching/S836/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S836 (pre-registered in knowledge/DECISIONS.md before any S836 data): where is the END-specific information that the late lookup's queries u |
| [S837](../experiments/05_query_patching/S837/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S837 (pre-registered in knowledge/DECISIONS.md before any S837 data): which components write the END-specific information that the late look |
| [S838](../experiments/05_query_patching/S838/) | 0 | 0 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 |  |
| [S839](../experiments/05_query_patching/S839/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S839 (pre-registered in knowledge/DECISIONS.md before any S839 data): gaps 1 and 2 of the user's list (10-17) - what is the END reading mode |
| [S842](../experiments/05_query_patching/S842/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S842 (pre-registered in knowledge/DECISIONS.md before any S842 data): gap 1 continued - where in the non-address part of the queries is the  |
| [S844](../experiments/05_query_patching/S844/) | 2 | 4 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S844 (pre-registered in knowledge/DECISIONS.md before any S844 data; user 10-17): separate the predecessor bias of the late update from goal |
| [S847](../experiments/05_query_patching/S847/) | 2 | 3 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S847 (pre-registered in knowledge/DECISIONS.md before any S847 data; continues S845 after its PATH near miss): does the band's correction re |
| [S856](../experiments/05_query_patching/S856/) | 2 | 7 | circuit v5, circuit v6, circuit v7, figure/table builders | S856 (P1, user 10-18; pre-registered in knowledge/DECISIONS.md before any S856 data). Does the visual-prefix-conditioned late query carry th |
| [S856b](../experiments/05_query_patching/S856b/) | 2 | 1 | circuit v5, circuit v6, circuit v7, figure/table builders | S856b (rerun of S856 after its reference check missed by one run; pre-registered in knowledge/DECISIONS.md before any S856b data). Change fr |

## 06_read_interchange

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S758](../experiments/06_read_interchange/S758/) | 5 | 7 | figure/table builders | S758 descriptive tables (exploratory, after the frozen analysis; not a registered verdict). Per arm: answer shares on 6 / 3 / 5 (literal of  |
| [S759](../experiments/06_read_interchange/S759/) | 3 | 4 | - | S759: where after layer 26 is the frame word read, and through which route? Complete-route read interchange of the frame word. Pre-registrat |
| [S769](../experiments/06_read_interchange/S769/) | 3 | 8 | R3 v3, circuit v6, circuit v7, figure/table builders | S769: the END(4) paradox - at N = 7 the end-relative target equals the literal image 4, so no arithmetic is needed, yet END(4) fails. Does t |
| [S770](../experiments/06_read_interchange/S770/) | 2 | 5 | figure/table builders | S770: which part of the full-attention read at L35-59 carries the frame effect (S768's missing third) and the late query control of START an |
| [S772](../experiments/06_read_interchange/S772/) | 3 | 12 | figure/table builders | S772: faithfulness of the END circuit on new scenes, both image counts and a second END phrasing. Pre-registration in knowledge/DECISIONS.md |
| [S826](../experiments/06_read_interchange/S826/) | 2 | 2 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S826 (pre-registered in knowledge/DECISIONS.md before any S826 data): where is the stated count read, and which rows hold it? FINDINGS §916  |
| [S826b](../experiments/06_read_interchange/S826b/) | 2 | 2 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S826 (pre-registered in knowledge/DECISIONS.md before any S826 data): where is the stated count read, and which rows hold it? FINDINGS §916  |

## 07_component_patching

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S797](../experiments/07_component_patching/S797/) | 3 | 3 | - | S797 (pre-registered in knowledge/DECISIONS.md before any S797 data): which component writes the address? S795 placed the formation of the l |
| [S840](../experiments/07_component_patching/S840/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S840 (pre-registered in knowledge/DECISIONS.md before any S840 data): S837 found no single band component (layers 46-49) whose direct path t |
| [S845](../experiments/07_component_patching/S845/) | 2 | 3 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S845 (pre-registered in knowledge/DECISIONS.md before any S845 data; user 10-17): close the writer -> reader path. Writers = the token mixer |
| [S846](../experiments/07_component_patching/S846/) | 2 | 5 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S846 (pre-registered in knowledge/DECISIONS.md before any S846 data; user 10-17 final question 'why does the update not stop at p = j*?'). T |
| [S848](../experiments/07_component_patching/S848/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S848 (pre-registered in knowledge/DECISIONS.md before any S848 data; user 10-17 15:03, test A + development data for test B). Test A, part 1 |
| [S849](../experiments/07_component_patching/S849/) | 2 | 4 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S849-P (test B evaluation; registered in knowledge/DECISIONS.md before any S849 data, with the frozen readout and thresholds set on the S848 |
| [S849p](../experiments/07_component_patching/S849p/) | 1 | 1 | figure/table builders | S849-P analysis (rules fixed in knowledge/DECISIONS.md before any S849 data; thresholds and the readout fixed on the S848 development data). |
| [S849u](../experiments/07_component_patching/S849u/) | 1 | 2 | figure/table builders | S849-U analysis (rules fixed in knowledge/DECISIONS.md before any S849 data). Test A part 2: is the goal-attainment signal found by S848 use |
| [S850](../experiments/07_component_patching/S850/) | 2 | 5 | circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S850 (pre-registered in knowledge/DECISIONS.md before any S850 data; user 10-17 evening): boundary-level stop-signal causal test - can the h |
| [S851](../experiments/07_component_patching/S851/) | 2 | 4 | circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S851 (pre-registered in knowledge/DECISIONS.md before any S851 data; user 10-17 late evening, minimal path test after S850): do the late ful |
| [S852](../experiments/07_component_patching/S852/) | 2 | 2 | circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S852 (pre-registered in knowledge/DECISIONS.md before any S852 data; user 10-18 'continue' after the S851 report): how is the wrong late que |
| [S853](../experiments/07_component_patching/S853/) | 2 | 3 | circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S853 (pre-registered in knowledge/DECISIONS.md before any S853 data; user 10-18 'GO - writer input decomposition'). What does the writer (la |
| [S854b](../experiments/07_component_patching/S854b/) | 2 | 2 | figure/table builders | S854-2 (pre-registered in knowledge/DECISIONS.md before any S854-2 data; user 10-18 'S854-2 - input-row causal decomposition of the writer') |

## 08_deltanet_source_cut

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S703](../experiments/08_deltanet_source_cut/S703/) | 2 | 2 | figure/table builders | S703: can the end of the image sequence be moved by hiding the last images from one module only? (the end-moving table) Pre-registration in  |
| [S706](../experiments/08_deltanet_source_cut/S706/) | 2 | 2 | figure/table builders | S706: do the anchor heads hand the end of the image sequence to the matcher heads' queries? (sender -> receiver, Cheng et al. style) Pre-reg |
| [S707](../experiments/08_deltanet_source_cut/S707/) | 2 | 2 | figure/table builders | S707: which heads read the end at L43 / L47 -- the matcher heads themselves, or the other heads? Pre-registration in knowledge/DECISIONS.md  |
| [S748](../experiments/08_deltanet_source_cut/S748/) | 2 | 3 | circuit v6, circuit v7, figure/table builders | S748: is the end that Qwen3.5-27B counts back from the last block's code or the number of visible images? Pre-registration in knowledge/DECI |
| [S857](../experiments/08_deltanet_source_cut/S857/) | 2 | 2 | circuit v5, circuit v6, circuit v7, figure/table builders | S857 (P2, user 10-18 plan; pre-registered in knowledge/DECISIONS.md before any S857 data). How do pictures 7-8 reach the late query? Same sc |

## 09_count_pathway

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S854a](../experiments/09_count_pathway/S854a/) | 2 | 2 | figure/table builders | S854-1 (pre-registered in knowledge/DECISIONS.md before any S854-1 data; user 10-18 'GO - first separate the ordinal from the image counts') |
| [S855](../experiments/09_count_pathway/S855/) | 2 | 4 | circuit v5, circuit v6, circuit v7, figure/table builders | S855 (Gap 2, second stage; pre-registered in knowledge/DECISIONS.md before any S855 data). Does the actual-count dependence of the late look |

## 10_r3_rebinding

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S601](../experiments/10_r3_rebinding/S601/) | 0 | 0 | R3 (Oct 6), figure/table builders |  |
| [S601h](../experiments/10_r3_rebinding/S601h/) | 0 | 0 | R3 (Oct 6) |  |
| [S609c](../experiments/10_r3_rebinding/S609c/) | 2 | 0 | R3 (Oct 6), figure/table builders | S609 (3): the end reader at the site where the model carries the image index independently of the prompt layout (S608). Fitted on images-fir |
| [S616b](../experiments/10_r3_rebinding/S616b/) | 1 | 7 | R3 (Oct 6), figure/table builders | S616b: choose the write layer inside the model by forward rebinding efficacy, then rebuild S609c and evaluate (pre-registered, exp/QUEUE.md  |
| [S619g](../experiments/10_r3_rebinding/S619g/) | 0 | 0 | R3 (Oct 6), figure/table builders |  |
| [S641](../experiments/10_r3_rebinding/S641/) | 0 | 0 | R3 (Oct 6) |  |
| [S709](../experiments/10_r3_rebinding/S709/) | 3 | 3 | - | S709-cal: model-internal R³ readers for Qwen3.5-27B under the existing R³ calibration protocol (no end-relative answers anywhere). (1) End r |
| [S710](../experiments/10_r3_rebinding/S710/) | 4 | 4 | figure/table builders | S710: does the R³ gain on 27B travel through the matcher heads' queries? (repair mediation) Pre-registration in knowledge/DECISIONS.md befor |
| [S711](../experiments/10_r3_rebinding/S711/) | 2 | 2 | figure/table builders | S711: R³-v2 -- the one change the 27B mechanism asks for: an ordinal bound to an END frame always gets the frame write (END -> START), also  |
| [S712](../experiments/10_r3_rebinding/S712/) | 2 | 3 | figure/table builders | S712: independent procedural check of R³-v2 on Qwen3.5-27B before natural benchmarks -- other scene sizes, prompt layouts and end words than |
| [S713](../experiments/10_r3_rebinding/S713/) | 2 | 108 | - | S713: R³-v2 with the 27B model-internal readers on natural benchmarks, through the unchanged S582 harness (exp/s582_evalrepair.run: same ite |
| [S714](../experiments/10_r3_rebinding/S714/) | 2 | 2 | figure/table builders | S714: the k = N boundary route of Qwen3.5-27B -- how "the N-th image from the last" (= the first image) is answered although it does not use |
| [S715](../experiments/10_r3_rebinding/S715/) | 2 | 5 | - | S715-diag: why R³-v2's ordinal detector misses option-line ordinals on MMIU / MuirBench (27B) -- a READ-stage measurement on the first-use ( |
| [S716](../experiments/10_r3_rebinding/S716/) | 1 | 1 | - | S716 analysis, frozen with the pre-registration (knowledge/DECISIONS.md, 2026-10-09) before any S716 (R³-v3) data. R³-v3 = R³-v2 with the RE |
| [S717](../experiments/10_r3_rebinding/S717/) | 2 | 0 | figure/table builders | S717-cal: R³ frame anchors for other end-relative wordings (frame identification only -- no end-relative answers anywhere). R³-v3 binds an o |
| [S721](../experiments/10_r3_rebinding/S721/) | 2 | 3 | circuit v6, circuit v7, figure/table builders | S721 latency / memory: per-input wall time of the unedited forward vs R³ (capture forward + plan + repaired forward when something is writte |
| [S724](../experiments/10_r3_rebinding/S724/) | 4 | 25 | figure/table builders | S724 plan-level evaluation (amendment before any S724 data; user direction 2026-10-10: "개별 부품 선택 규칙은 유지하되, 완성된 쓰기 계획의 정확도를 별도의 보정용 검증 자료에서 확 |
| [S729](../experiments/10_r3_rebinding/S729/) | 2 | 12 | figure/table builders | S729: re-calibrate the end reader (READ's count part) over image resolutions -- block-position labels only, no answer, no CIFAR. The S609 /  |
| [S730](../experiments/10_r3_rebinding/S730/) | 5 | 26 | figure/table builders | S730: why R³ works, stage by stage, inside each model -- the index-matching heads' queries and the answer under five conditions. Pre-registr |
| [S731](../experiments/10_r3_rebinding/S731/) | 1 | 0 | - | S731: write vectors at the model's rule write layer L_w (S623(b) forward-only consistency), read layer unchanged. Ordinal prototypes and fra |
| [S732](../experiments/10_r3_rebinding/S732/) | 0 | 0 | - |  |
| [S733](../experiments/10_r3_rebinding/S733/) | 2 | 2 | figure/table builders | S733: other model families -- the earlier modules' calibrations (S616b: InternVL3-8B, Qwen2.5-VL-7B, LLaVA-OneVision-7B) in the R3MinMD form |
| [S733b](../experiments/10_r3_rebinding/S733b/) | 1 | 1 | figure/table builders | S733 with the amended reader-admission rule (knowledge/DECISIONS.md, 2026-10-11 12:50): a copy of exp/s733_analyze.py with the same rules an |
| [S734](../experiments/10_r3_rebinding/S734/) | 0 | 2 | figure/table builders |  |
| [S736](../experiments/10_r3_rebinding/S736/) | 2 | 11 | circuit v6, circuit v7, figure/table builders | S736: when is the frame write needed, and is its direction what matters? (pre-registration knowledge/DECISIONS.md 2026-10-10 16:35) The S730 |
| [S737](../experiments/10_r3_rebinding/S737/) | 2 | 7 | figure/table builders | S737: the option-line query on natural images (pre-registration knowledge/DECISIONS.md 2026-10-10, before any S737 data). Items: the MMIU de |
| [S738](../experiments/10_r3_rebinding/S738/) | 2 | 2 | figure/table builders | S738 (1): is the R³ calibration set unrelated to every evaluation set? Perceptual-hash audit, no model call. Calibration images = the natura |
| [S738b](../experiments/10_r3_rebinding/S738b/) | 1 | 2 | figure/table builders | S738b: the registered S724 plan analysis (exp/s724_plan.py main) on the selection split and the de-duplicated confirmation split, written to |
| [S745](../experiments/10_r3_rebinding/S745/) | 2 | 4 | figure/table builders | S745: the core causal tests in a famous VLM with a non-Qwen language model, without R³ (reply to the mock reviews; the user: "LLaVA 말고 선행 연구 |
| [S745b](../experiments/10_r3_rebinding/S745b/) | 1 | 1 | figure/table builders | S745b analysis (amended after the S745 gate, before any main-stage data; knowledge/DECISIONS.md 2026-10-11 20:29): a copy of exp/s745_analyz |
| [S791](../experiments/10_r3_rebinding/S791/) | 2 | 6 | R3 v3, circuit v6, circuit v7, figure/table builders | S791 (pre-registered in knowledge/DECISIONS.md before any S791 data): one causal chain in the same scenes -- address / frame writes -> the l |
| [S833](../experiments/10_r3_rebinding/S833/) | 2 | 2 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S833 (pre-registered in knowledge/DECISIONS.md before any S833 data): does R3's frame rebind work by turning off END's late re-lookup? Phase |

## 11_benchmarks

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S582](../experiments/11_benchmarks/S582/) | 1 | 97 | R3 (Oct 6), figure/table builders | S582: one evaluation for every repair -- the base model and the repaired model on the SAME items, paired. Repairs: lora (S579 circuit-band a |

## 12_laws_and_predictions

| ID | code | data | papers | summary |
|---|---:|---:|---|---|
| [S832](../experiments/12_laws_and_predictions/S832/) | 2 | 3 | circuit v2, circuit v3, circuit v4, circuit v5, circuit v6, circuit v7, figure/table builders | S832 (pre-registered in knowledge/DECISIONS.md before any S832 data): does the two-component law FROZEN after S831 (research/ces/outputs/ s8 |
| [S841](../experiments/12_laws_and_predictions/S841/) | 2 | 2 | circuit v3, circuit v4, circuit v5, circuit v6, circuit v7 | S841 (pre-registered in knowledge/DECISIONS.md before any S841 data): why does the late step move the identity reference? Hypothesis (sign): |
| [S858](../experiments/12_laws_and_predictions/S858/) | 2 | 3 | circuit v5, circuit v6, circuit v7, figure/table builders | S858 (P3, user 10-18: scope frozen after S857; pre-registered in knowledge/DECISIONS.md before any S858 data). Prior predictions of what unp |
