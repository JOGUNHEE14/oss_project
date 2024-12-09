import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

class EmotionPredict:
    """
    The EmotionPredictor class is used to predict the emotion of a given text
    and return the corresponding emoji.

    Parameters
    ----------
    model_path : str, optional
        Path to the trained model file. Default is 'emotion_mlp_model.sav' within the package.
    vectorizer_path : str, optional
        Path to the CountVectorizer file. Default is 'emotion_vectorizer.pkl' within the package.
    label_encoder_path : str, optional
        Path to the LabelEncoder file. Default is 'emotion_label_encoder.pkl' within the package.
    """
    
    def __init__(self, model_path=None, vectorizer_path=None, label_encoder_path=None):
        """
        Initializes the EmotionPredictor instance.

        Parameters
        ----------
        model_path : str, optional
            Path to the trained model file. Default is 'emotion_mlp_model.sav' within the package.
        vectorizer_path : str, optional
            Path to the CountVectorizer file. Default is 'emotion_vectorizer.pkl' within the package.
        label_encoder_path : str, optional
            Path to the LabelEncoder file. Default is 'emotion_label_encoder.pkl' within the package.
        
        Raises
        ------
        FileNotFoundError
            Raised if the specified file does not exist.
        """
        base_path = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_path, 'emotion_mlp_model.sav')
        self.vectorizer_path = vectorizer_path or os.path.join(base_path, 'emotion_vectorizer.pkl')
        self.label_encoder_path = label_encoder_path or os.path.join(base_path, 'emotion_label_encoder.pkl')

        for path, name in [(self.model_path, "model"), (self.vectorizer_path, "vectorizer"), (self.label_encoder_path, "label encoder")]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{name} file does not exist: {path}")

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        with open(self.vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.label_encoder_path, 'rb') as f:
            self.label_encoder = pickle.load(f)

    def predict(self, text: str) -> str:
        """
        Predicts the emotion of the input text and returns the corresponding emoji.

        Parameters
        ----------
        text : str
            Text for which to predict the emotion.

        Returns
        -------
        str
            Predicted emotion and corresponding emoji.
        
        Examples
        --------
        >>> predictor = EmotionPredictor()
        >>> result = predictor.predict("I am so happy today!")
        >>> print(result)
        Predicted emotion: Joy, Emoji: 😊
        """
        emotion_to_emoji = {
            "Anger": "😡",
            "Joy": "😊",
            "Anxiety": "😰",
            "Surprise": "😳",
            "Sadness": "😢",
            "Hurt": "💔"
        }

        input_vector = self.vectorizer.transform([text])
        predicted_label = self.model.predict(input_vector)
        emotion = self.label_encoder.inverse_transform(predicted_label)[0]
        emoji = emotion_to_emoji.get(emotion, "❓")

        return f"Predicted emotion: {emotion}, Emoji: {emoji}"

