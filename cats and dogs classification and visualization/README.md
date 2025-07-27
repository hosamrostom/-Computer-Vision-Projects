# 🐱👕 Cats vs Clothes Classification using CNN & VGG16

This project focuses on building image classification models to distinguish between images of **cats** and **clothing items** using:

- A **custom Convolutional Neural Network (CNN)** model built from scratch  
- A **pre-trained VGG16** model fine-tuned on the same dataset  
- Filter visualization to understand how the models "see" the data

---

## 📁 Dataset

The dataset consists of two classes:
- `cats/`: contains images of domestic cats
- `clothes/`: contains various clothing item images

Images were resized to `(150, 150, 3)`.

---

## 🧠 Models

### 1. Custom CNN (from scratch)
- 3 Convolutional layers + MaxPooling
- Flatten → Dense → Dropout → Output (sigmoid)

```python
model = Sequential([
  Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
  MaxPooling2D(2,2),
  Conv2D(64, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Conv2D(128, (3,3), activation='relu'),
  MaxPooling2D(2,2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(1, activation='sigmoid')
])


 Author
Developed by: [Hossam Mustafa ]
Passionate about CV 
.
Feel free to connect or contribute!

