import os
from textwrap import dedent

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv(".env")

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "La variabile OPENAI_API_KEY non è presente nel file .env"
    )

client = OpenAI()

MODEL = "gpt-4o"

# ESEMPIO 1: TUTOR MATEMATICO

math_tutor_prompt = """
    Sei un tutor di matematica.
    Ti verrà fornito un problema matematico.

    Devi fornire una soluzione passo dopo passo e una risposta finale.

    Per ogni passaggio:
    - explanation contiene la spiegazione;
    - output contiene l'operazione o l'equazione.
"""


class MathReasoning(BaseModel):

    class Step(BaseModel):
        explanation: str
        output: str

    steps: list[Step]
    final_answer: str


def get_math_solution(question: str) -> MathReasoning:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": dedent(math_tutor_prompt),
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        response_format=MathReasoning,
    )

    message = completion.choices[0].message

    if message.refusal:
        raise ValueError(f"Richiesta rifiutata: {message.refusal}")

    if message.parsed is None:
        raise ValueError("Non è stato restituito un risultato strutturato.")

    return message.parsed


def print_math_solution(result: MathReasoning) -> None:
    print("\n" + "=" * 60)
    print("ESEMPIO 1: TUTOR MATEMATICO")
    print("=" * 60)

    for numero, step in enumerate(result.steps, start=1):
        print(f"\nPassaggio {numero}")
        print(f"Spiegazione: {step.explanation}")
        print(f"Operazione: {step.output}")

    print(f"\nRisposta finale: {result.final_answer}")

# ESEMPIO 2: RICETTA

recipe_prompt = """
    Analizza la descrizione della ricetta fornita dall'utente.

    Restituisci:
    - il nome della ricetta;
    - il numero di porzioni;
    - il tempo di preparazione in minuti;
    - la lista degli ingredienti;
    - la lista dei passaggi;
    - indica se la ricetta è senza glutine.
"""


class Recipe(BaseModel):

    class Ingredient(BaseModel):
        name: str
        quantity: str

    recipe_name: str
    servings: int
    preparation_minutes: int
    ingredients: list[Ingredient]
    steps: list[str]
    gluten_free: bool


def get_recipe(text: str) -> Recipe:
    completion = client.beta.chat.completions.parse(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": dedent(recipe_prompt),
            },
            {
                "role": "user",
                "content": text,
            },
        ],
        response_format=Recipe,
    )

    message = completion.choices[0].message

    if message.refusal:
        raise ValueError(f"Richiesta rifiutata: {message.refusal}")

    if message.parsed is None:
        raise ValueError("Non è stato restituito un risultato strutturato.")

    return message.parsed


def print_recipe(recipe: Recipe) -> None:
    print("\n" + "=" * 60)
    print("ESEMPIO 2: RICETTA")
    print("=" * 60)

    print(f"\nNome: {recipe.recipe_name}")
    print(f"Porzioni: {recipe.servings}")
    print(f"Preparazione: {recipe.preparation_minutes} minuti")
    print(f"Senza glutine: {recipe.gluten_free}")

    print("\nIngredienti:")

    for ingredient in recipe.ingredients:
        print(f"- {ingredient.name}: {ingredient.quantity}")

    print("\nProcedimento:")

    for numero, step in enumerate(recipe.steps, start=1):
        print(f"{numero}. {step}")

# AVVIO DEL PROGRAMMA

if __name__ == "__main__":
    try:
        math_result = get_math_solution(
            "Come posso risolvere l'equazione 8x + 7 = -23?"
        )

        print_math_solution(math_result)

        recipe_result = get_recipe(
            """
            Prepara dei pancake per due persone usando:
            120 grammi di farina di riso,
            150 millilitri di latte,
            un uovo,
            un cucchiaino di lievito senza glutine
            e un cucchiaino di zucchero.

            Mescola gli ingredienti e cuoci i pancake
            in padella per circa 15 minuti.
            """
        )

        print_recipe(recipe_result)

    except Exception as error:
        print(f"\nSi è verificato un errore: {error}")