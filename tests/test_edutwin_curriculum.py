from pathlib import Path

from week1_llm.curriculum import CurriculumCatalog


def test_catalog_discovers_grade_subject_topic_from_knowledge_base(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "grade8_mathematics_algebra.txt").write_text(
        "GRADE 8 MATHEMATICS\nSUBJECT: MATHEMATICS\nTOPIC: ALGEBRA\n"
        "Solving equations with brackets.",
        encoding="utf-8",
    )
    catalog = CurriculumCatalog(knowledge)

    assert catalog.grades == [f"Grade {n}" for n in range(8, 13)]
    assert catalog.subjects_for_grade("Grade 8") == ["Mathematics"]
    assert catalog.topics_for("Grade 8", "Mathematics") == ["Algebra"]
    assert catalog.sources_for("Grade 8", "Mathematics", "Algebra") == [
        "grade8_mathematics_algebra.txt"
    ]


def test_catalog_does_not_invent_subjects_for_missing_grade(tmp_path):
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "grade8_mathematics_algebra.txt").write_text(
        "GRADE 8 MATHEMATICS\nSUBJECT: MATHEMATICS\nTOPIC: ALGEBRA\nContent",
        encoding="utf-8",
    )
    catalog = CurriculumCatalog(knowledge)

    assert catalog.subjects_for_grade("Grade 12") == []
    assert not catalog.has_curriculum("Grade 12", "Mathematics", "Algebra")
