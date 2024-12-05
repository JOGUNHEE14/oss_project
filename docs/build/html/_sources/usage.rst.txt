Creating recipes
----------------

To analyze the emotion contained in a sentence in English,
you can use ``pymood.anlaysis_emotion()`` function
This function analyzes the mood of sentences collected in English and 
returns emoticons that fit the situation.

.. py:function:: pymood.anlaysis_emotion(kind = String)

   Return a emoji.

   Prameters
   ---------
   **Kind** : str
        Enter the sentence to analyze the emotion of the string type.
        entences should be in English.

   Returns
   -------
   **str**
        emoji kinds:

        - 😊 : Happiness  

        - 😢 : Sadness  

        - 😴 : Sleepiness or boredom  

        - 😰 : Anxiety or nervousness  

        - 😲 : Surprise  

        - 😡 : Anger  

        - 😌 : Calmness or relief  

        - 😔 : Disappointment  

        - 😩 : Exhaustion or frustration  

        - 😍 : Love or affection


