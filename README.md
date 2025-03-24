# Custom-Transformer-Based-LLM-using-Python

The project involves designing a custom Transformer-based Language Model (LLM) from scratch for text summarization. Key components and features implemented include:
Architecture:

o	Encoder-decoder structure with 6 layers for both the encoder and decoder.
o	Multi-head attention mechanism with 8 heads for effective representation learning.
o	Positional encoding to preserve the sequence order of input text.
o	Position-wise feed-forward networks with ReLU activation for feature transformation.


Dataset:
•	The SAMSum dataset was used, consisting of dialogues and their corresponding summaries.
•	Preprocessing: 
o	Tokenization: Converts text into sequences of tokens.
o	Padding: Ensures all sequences have a uniform length.
o	Batching: Efficiently processes multiple sequences in parallel during training.


Training the Model
Optimizer and Hyperparameters:
•	
o	Adam optimizer was used with a learning rate of 0.0005.
o	Batch size: 16
o	Epochs: 8
•	
Techniques to Prevent Overfitting:
o	Early stopping based on validation loss.
o	Regular dropout applied to layers.
Performance during Training:
o	Training loss decreased from 3.12 (Epoch 1) to 0.95 (Epoch 8).
o	Validation loss decreased from 2.85 (Epoch 1) to 0.91 (Epoch 8).

Conclusion
The Transformer-based LLM for text summarization achieved strong results, as indicated by the ROUGE scores and qualitative evaluation of generated summaries. The model effectively leverages self-attention mechanisms and positional encoding to summarize long texts.
![image](https://github.com/user-attachments/assets/bf4c58cd-5b2d-4945-bc17-cc17db14cbef)
![image](https://github.com/user-attachments/assets/09c8691e-3768-446a-bf68-387823aade52)
![image](https://github.com/user-attachments/assets/5082a41f-f3b0-4c4f-8e73-569c29215a14)
![image](https://github.com/user-attachments/assets/ff55e5c3-b4b7-4bfc-ad96-2f6a447e62c8)


