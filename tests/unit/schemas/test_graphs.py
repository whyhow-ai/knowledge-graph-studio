# import pytest
# from bson import ObjectId
# from pydantic import ValidationError

# from whyhow_api.schemas.graphs import CreateGraphBody


# class TestCreateGraphBody:
#     """Test GraphCreate for graph."""

#     def test_correct_values(self):
#         """Test correct values."""
#         workspace = str(ObjectId())
#         schema_ = str(ObjectId())
#         questions = ["What is the capital of France?"]
#         seed_concept = "Paris"

#         question_body = CreateGraphBody(
#             name="Test Graph",
#             workspace=workspace,
#             questions=questions,
#             seed_concept=seed_concept,
#             filters={},
#         )
#         assert question_body.name == "Test Graph"
#         assert question_body.workspace == ObjectId(workspace)
#         assert question_body.questions == questions
#         assert question_body.seed_concept == seed_concept
#         assert question_body.filters == {}

#         schema_body = CreateGraphBody(
#             name="Test Graph",
#             workspace=workspace,
#             schema_=schema_,
#         )

#         assert schema_body.name == "Test Graph"
#         assert schema_body.workspace == ObjectId(workspace)
#         assert schema_body.schema_ == ObjectId(schema_)

#     @pytest.mark.parametrize(
#         "name, workspace, schema_, questions, filters",
#         [
#             (None, None, None, None, None),  # None values
#             (
#                 None,
#                 ObjectId(),
#                 ObjectId(),
#                 None,
#                 None,
#             ),  # None name
#             ("Test Graph", None, ObjectId(), None, None),  # None workspace
#             ("Test Graph", ObjectId(), None, None, None),  # None schema
#             (
#                 "Test Graph",
#                 ObjectId(),
#                 ObjectId(),
#                 None,
#                 {"invalid": []},
#             ),  # Invalid filter
#             (
#                 "Test Graph",
#                 ObjectId(),
#                 ObjectId(),
#                 ["Question 1"],
#                 None,
#             ),  # Questions and schema provided
#         ],
#     )
#     def test_validation_errors(
#         self, name, workspace, schema_, questions, filters
#     ):
#         with pytest.raises(ValidationError):
#             CreateGraphBody(
#                 name=name,
#                 workspace=workspace,
#                 schema_=schema_,
#                 filters=filters,
#                 questions=questions,
#             )
