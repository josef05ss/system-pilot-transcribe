from app.services.transcriber import deduplicate_overlap


def test_deduplicates_repeated_words():
    previous = "hoy revisaremos el modelo relacional y sus componentes"
    current = "el modelo relacional y sus componentes antes de continuar"
    assert deduplicate_overlap(previous, current) == "antes de continuar"
