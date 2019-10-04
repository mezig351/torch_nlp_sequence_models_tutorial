torch Natural Language Processing Tutorial Solutions:
---

Update: Now there is Tutorial 5 solutions too, batched and/or combined with tutorial 4; and the notebooks for the previous tutorials. 
To speed up, I mean batch, tutorial 5 I have used Lattice LSTM, credit to https://github.com/YuxueShi/LatticeLSTM 

My solution to tutorial 4 from pytorch's official tutorial series, Deep Learning for NLP with Pytorch -Sequence Models and Long-Short Term Memory Networks

https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#exercise-augmenting-the-lstm-part-of-speech-tagger-with-character-level-features

There are two solutions, one uses a for loop to go from characters to words (char_emb_lstm.py), and the other one uses the pack_padded_sequences function from rnn utils (char_emb_lstm_packed.py).

Feel free to comment!