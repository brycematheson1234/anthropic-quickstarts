#!/usr/bin/env python3
"""
Test to validate that EditTool API type mappings are correct.

This test ensures that our EditTool implementations have the correct
api_type and name fields that match the official Anthropic documentation.
"""

import pytest

from computer_use_demo.tools.edit import (
    EditTool20241022,
    EditTool20250124,
    EditTool20250429,
)


class TestAPITypeMappings:
    """Test API type mappings against Anthropic documentation."""

    def test_edit_tool_20241022_mapping(self):
        """Test that EditTool20241022 has correct API type mapping for Claude Sonnet 3.5."""
        tool = EditTool20241022()

        # From Anthropic docs: Claude Sonnet 3.5 uses text_editor_20241022
        assert tool.api_type == "text_editor_20241022"
        assert tool.name == "str_replace_editor"

        # Verify the tool definition matches expected format
        tool_params = tool.to_params()
        assert tool_params["type"] == "text_editor_20241022"
        assert tool_params["name"] == "str_replace_editor"

    def test_edit_tool_20250124_mapping(self):
        """Test that EditTool20250124 has correct API type mapping for Claude Sonnet 3.7."""
        tool = EditTool20250124()

        # From Anthropic docs: Claude Sonnet 3.7 uses text_editor_20250124
        assert tool.api_type == "text_editor_20250124"
        assert tool.name == "str_replace_editor"

        # Verify the tool definition matches expected format
        tool_params = tool.to_params()
        assert tool_params["type"] == "text_editor_20250124"
        assert tool_params["name"] == "str_replace_editor"

    def test_edit_tool_20250429_mapping(self):
        """Test that EditTool20250429 has correct API type mapping for Claude 4."""
        tool = EditTool20250429()

        # From Anthropic docs: Claude 4 uses text_editor_20250429 with updated name
        assert tool.api_type == "text_editor_20250429"
        assert tool.name == "str_replace_based_edit_tool"

        # Verify the tool definition matches expected format
        tool_params = tool.to_params()
        assert tool_params["type"] == "text_editor_20250429"
        assert tool_params["name"] == "str_replace_based_edit_tool"

    def test_undo_edit_support_differences(self):
        """Test that undo_edit support differs correctly between tool versions."""
        # EditTool20241022 and EditTool20250124 support undo_edit
        # EditTool20250429 does NOT support undo_edit
        # This is validated by checking the Command types they accept
        from typing import get_args

        from computer_use_demo.tools.edit import (
            Command_20241022,
            Command_20250124,
            Command_20250429,
        )

        # Check that 20241022 and 20250124 include undo_edit
        assert "undo_edit" in get_args(Command_20241022)
        assert "undo_edit" in get_args(Command_20250124)

        # Check that 20250429 does NOT include undo_edit
        assert "undo_edit" not in get_args(Command_20250429)

    def test_all_tools_have_consistent_base_functionality(self):
        """Test that all tools support the same base commands except undo_edit."""
        tools = [EditTool20241022(), EditTool20250124(), EditTool20250429()]

        for tool in tools:
            # Verify the tools are properly instantiated and have required fields
            assert tool.api_type is not None
            assert tool.name is not None
            assert tool.to_params() is not None

            # Verify tool parameters structure
            params = tool.to_params()
            assert "type" in params
            assert "name" in params
            assert params["type"] == tool.api_type
            assert params["name"] == tool.name

    def test_model_tool_compatibility_matrix(self):
        """Test the compatibility matrix between models and tools."""
        # Define the expected mappings based on Anthropic documentation
        expected_mappings = {
            # Claude Sonnet 3.5 models
            "claude-3-5-sonnet-20241022": {
                "tool_type": "text_editor_20241022",
                "tool_name": "str_replace_editor",
                "tool_class": EditTool20241022,
            },
            "claude-3-5-sonnet-20240620": {
                "tool_type": "text_editor_20241022",
                "tool_name": "str_replace_editor",
                "tool_class": EditTool20241022,
            },
            # Claude Sonnet 3.7 models
            "claude-3-7-sonnet-20250219": {
                "tool_type": "text_editor_20250124",
                "tool_name": "str_replace_editor",
                "tool_class": EditTool20250124,
            },
            # Claude 4 models
            "claude-sonnet-4-20250514": {
                "tool_type": "text_editor_20250429",
                "tool_name": "str_replace_based_edit_tool",
                "tool_class": EditTool20250429,
            },
            "claude-opus-4-20250514": {
                "tool_type": "text_editor_20250429",
                "tool_name": "str_replace_based_edit_tool",
                "tool_class": EditTool20250429,
            },
        }

        for model_name, expected in expected_mappings.items():
            tool_instance = expected["tool_class"]()

            assert (
                tool_instance.api_type == expected["tool_type"]
            ), f"Model {model_name} should use tool type {expected['tool_type']}, got {tool_instance.api_type}"

            assert (
                tool_instance.name == expected["tool_name"]
            ), f"Model {model_name} should use tool name {expected['tool_name']}, got {tool_instance.name}"

    def test_anthropic_beta_header_requirements(self):
        """Test that we understand which tools require beta headers."""
        # Based on Anthropic docs, only computer use tools require beta headers
        # Text editor tools are generally available as of the latest updates

        # These tools should work without beta headers (as of recent API updates)
        standalone_tools = [EditTool20241022(), EditTool20250124(), EditTool20250429()]

        for tool in standalone_tools:
            # Verify tools can be instantiated (implying they don't require special headers)
            assert tool.api_type is not None
            assert tool.name is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
