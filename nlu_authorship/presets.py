"""Curated passage pairs sampled directly from the locked benchmark test split."""
from __future__ import annotations

DEMO_PRESETS = [
    {
        "id": "imdb_same",
        "title_en": "Short Film Reviews (Same Author)",
        "title_ko": "단문 영화 리뷰 (동일 저자)",
        "domain": "imdb_reviews",
        "stratum": "open_unseen_author_same_topic",
        "ground_truth": "same_author",
        "pair_id": 2443,
        "text_1": (
            "There has been too many cop films , but I can explain why this one dies . "
            "Not a bad film , but it's a copy of Lethal Weapon formula ; of two different types of police partners . "
            "It's a did that , done it before type of film . With no surprises , and the typical big ending . "
            "It's well directed , and acted , but the story kills this one ."
        ),
        "text_2": (
            "I like history movies , so I like this one . Nice change for Van Damm he plays a different role . "
            "The film is interesting to follow . We have the reasons why people joined , and the harshness of being one . "
            "A solid 6 , and I would watch this film when they show it on TBS ."
        ),
    },
    {
        "id": "arxiv_same",
        "title_en": "Zero Shot ArXiv Physics (Same Author)",
        "title_ko": "제로샷 ArXiv 논문 초록 (동일 저자)",
        "domain": "arxiv_research",
        "stratum": "zero_shot_unseen_genre",
        "ground_truth": "same_author",
        "pair_id": 165,
        "text_1": (
            "We prove that a single photon with quantum data encoded in its orbital "
            "angular momentum can be manipulated with simple optical elements to provide any "
            "desired quantum computation. We will show how to build any quantum unitary "
            "operator using beamsplitters, phase shifters, holograms and an extraction gate "
            "based on quantum interrogation. The advantages and challenges of these approach "
            "are then discussed, in particular the problem of the readout of the results."
        ),
        "text_2": (
            "Inside computer networks, different information processing tasks are "
            "necessary to deliver the user data efficiently. This processing can also be "
            "done in the quantum domain. We present simple optical quantum networks where "
            "the orbital angular momentum of a single photon is used as an ancillary degree "
            "of freedom which controls decisions at the network level. Linear optical "
            "elements are enough to provide important network primitives like multiplexing "
            "and routing. First we show how to build a simple multiplexer and demultiplexer "
            "which combine photonic qubits and separate them again at the receiver. We also "
            "give two different self-routing networks where the OAM of an input photon is "
            "enough to make it find its desired destination."
        ),
    },
    {
        "id": "news_diff",
        "title_en": "Ultra Short News Snippets (Different Authors)",
        "title_ko": "초단문 뉴스 헤드라인 (다른 저자)",
        "domain": "news",
        "stratum": "open_unseen_author_same_topic",
        "ground_truth": "different_author",
        "pair_id": 942,
        "text_1": "Here's how Caitlyn Jenner responded when a friend asked for a date with Kendall Jenner",
        "text_2": (
            "Buffalo's Josh Allen had a rough start to his NFL career, similar to what the Jets' "
            "Zach Wilson is experiencing, and now the Bills quarterback is an MVP candidate."
        ),
    },
    {
        "id": "guardian_diff",
        "title_en": "Cross Genre Guardian Politics (Different Authors)",
        "title_ko": "교차 장르 Guardian 칼럼·서평 (다른 저자)",
        "domain": "guardian_journalism",
        "stratum": "cross_genre",
        "ground_truth": "different_author",
        "pair_id": 8990,
        "text_1": (
            "Here is a column about politics. On the latest research figures which guide BBC programming, "
            "that's 30% of you gone already. Constitutional politics. Bang goes another 20%. Local politics. "
            "Is there anybody still out there, mother? Certainly nobody under the age of 30, son. They think "
            "Have I Got News For You is pretty dense stuff. Well, never mind. The place where I live - "
            "population upwards of 230,000 - is having a referendum this week. Are we Middlesbrough or Brighton? "
            "Hartlepool or Sedgefield? Do we or don't we want an elected mayor? It used to be an issue at the "
            "cutting edge of New Labour's agenda. It arrives now, in Southwark, trailing clouds of the most "
            "perfect boredom, the most profound apathy. Few people I've talked to even know that the vote is "
            "coming. Voting no this Thursday isn't embracing the status quo. Voting no is voting for something better."
        ),
        "text_2": (
            "Bill Clinton had barely been unleashed upon America, when the mythology started to grow. In 1992 "
            "the Republicans insisted he was nothing more than the \"failed governor of a small southern state\", "
            "but Americans had a feeling he would soon be running their country so they wanted to know more. "
            "They gobbled up every morsel, from the story of Gennifer with a G to the legend that the young "
            "governor could eat an apple in a single bite. On the press bus we were no different. We heard that "
            "the candidate didn't snooze during the long-haul flights or late-night drives through the rural "
            "heartland. Once the train has pulled into San Francisco, and the 957th page has been read, what do "
            "you think of this man? And you realise, just from hearing about it afterwards, that it must have "
            "been a hell of a ride."
        ),
    },
]
