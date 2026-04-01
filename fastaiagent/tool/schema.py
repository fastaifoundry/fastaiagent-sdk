"""Schema validation and drift detection for tools."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class SchemaViolation(BaseModel):
    """A single schema violation found in tool output."""

    field: str
    expected_type: str
    actual_type: str
    message: str


class DriftReport(BaseModel):
    """Report of schema drift across multiple tool responses."""

    tool_name: str
    violations: list[SchemaViolation] = Field(default_factory=list)
    responses_checked: int = 0
    drift_detected: bool = False

    @property
    def summary(self) -> str:
        if not self.drift_detected:
            return (
                f"No drift detected for '{self.tool_name}' "
                f"across {self.responses_checked} responses"
            )
        return (
            f"Drift detected for '{self.tool_name}': "
            f"{len(self.violations)} violations across {self.responses_checked} responses"
        )


def _get_json_type(value: Any) -> str:
    """Get JSON type name for a Python value."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def validate_schema(
    schema: dict, response: Any, path: str = ""
) -> list[SchemaViolation]:
    """Validate a response against a JSON Schema and return violations."""
    violations: list[SchemaViolation] = []

    if not isinstance(schema, dict):
        return violations

    expected_type = schema.get("type")
    if expected_type and not _type_matches(expected_type, response):
        violations.append(
            SchemaViolation(
                field=path or "(root)",
                expected_type=expected_type,
                actual_type=_get_json_type(response),
                message=f"Expected {expected_type}, got {_get_json_type(response)}",
            )
        )
        return violations

    # Check object properties
    if expected_type == "object" and isinstance(response, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        for field_name in required:
            if field_name not in response:
                violations.append(
                    SchemaViolation(
                        field=f"{path}.{field_name}" if path else field_name,
                        expected_type=properties.get(field_name, {}).get("type", "any"),
                        actual_type="missing",
                        message=f"Required field '{field_name}' is missing",
                    )
                )

        for field_name, field_schema in properties.items():
            if field_name in response:
                field_path = f"{path}.{field_name}" if path else field_name
                violations.extend(
                    validate_schema(field_schema, response[field_name], field_path)
                )

        # Check for unexpected fields (additionalProperties)
        if schema.get("additionalProperties") is False:
            for key in response:
                if key not in properties:
                    violations.append(
                        SchemaViolation(
                            field=f"{path}.{key}" if path else key,
                            expected_type="none",
                            actual_type=_get_json_type(response[key]),
                            message=f"Unexpected field '{key}'",
                        )
                    )

    # Check array items
    if expected_type == "array" and isinstance(response, list):
        items_schema = schema.get("items")
        if items_schema:
            for i, item in enumerate(response):
                item_path = f"{path}[{i}]" if path else f"[{i}]"
                violations.extend(validate_schema(items_schema, item, item_path))

    return violations


def _type_matches(expected: str, value: Any) -> bool:
    """Check if a value matches the expected JSON Schema type."""
    type_checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
        "null": lambda v: v is None,
    }
    check = type_checks.get(expected)
    if check is None:
        return True
    return check(value)


def detect_drift(
    tool_name: str, output_schema: dict, responses: list[Any]
) -> DriftReport:
    """Detect schema drift across multiple tool responses."""
    all_violations: list[SchemaViolation] = []

    for response in responses:
        violations = validate_schema(output_schema, response)
        all_violations.extend(violations)

    return DriftReport(
        tool_name=tool_name,
        violations=all_violations,
        responses_checked=len(responses),
        drift_detected=len(all_violations) > 0,
    )
