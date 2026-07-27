from types import SimpleNamespace

from experience_os.input_resolver import ArtifactInputResolver


def task_fixture():
    return SimpleNamespace(
        user_scenario=SimpleNamespace(
            instructions=SimpleNamespace(
                known_info="Email: alice@example.com\nName: Alice Smith",
                task_instructions="Please use zip code 02139",
            )
        ),
        evaluation_criteria=SimpleNamespace(
            actions=[SimpleNamespace(arguments={"order_id": "#ORD123"})]
        ),
        initial_state=SimpleNamespace(
            initialization_actions=[SimpleNamespace(arguments={"user_id": "user_1234"})],
            initialization_data=SimpleNamespace(user_data={"unused": "value"}),
        ),
    )


def test_deterministic_candidates_from_task_sources():
    result = ArtifactInputResolver().resolve(task_fixture(), ["email", "first_name", "last_name", "zip", "order_id", "user_id"])
    assert result.params["email"] == "alice@example.com"
    assert result.params["first_name"] == "Alice"
    assert result.params["last_name"] == "Smith"
    assert result.params["zip"] == "02139"
    assert result.params["order_id"] == "#ORD123"
    assert result.params["user_id"] == "user_1234"
    assert not result.missing_fields
    assert not result.validation_errors


def test_schema_validation():
    schema = [
        {"name": "status", "type": "string", "enum": ["open", "closed"]},
        {"name": "email", "type": "string", "pattern": r"[^@]+@[^@]+\.[^@]+"},
    ]
    task = {"status": "bad", "email": "not-an-email"}
    result = ArtifactInputResolver().resolve(task, schema)
    assert any("enum" in error for error in result.validation_errors)
    assert any("pattern" in error for error in result.validation_errors)


def test_fake_llm_completes_missing_fields_and_is_validated():
    class FakeChat:
        def complete_json(self, messages):
            return {"email": "bob@example.com", "age": 4}

    resolver = ArtifactInputResolver(FakeChat())
    result = resolver.resolve({}, [{"name": "email", "type": "string"}, {"name": "age", "type": "integer"}], allow_llm=True)
    assert result.method == "deterministic+llm"
    assert result.params == {"email": "bob@example.com", "age": 4}
    assert not result.missing_fields
    assert not result.validation_errors
