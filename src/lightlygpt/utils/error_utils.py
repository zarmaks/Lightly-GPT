# Error handling utilities for the LightlyGPT application

import functools
import traceback

import streamlit as st


def handle_errors(func):
    """
    Decorator that wraps a function and catches exceptions,
    returning a standardized error message.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            # Log the full traceback for debugging
            traceback.print_exc()
            return error_msg

    return wrapper


def handle_tool_errors(func):
    """
    Decorator specifically for LangChain tools that catches exceptions
    and returns formatted error messages suitable for the agent.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_type = e.__class__.__name__
            error_msg = f"Tool Error ({error_type}): {str(e)}"
            # Log the full traceback for debugging
            traceback.print_exc()
            return error_msg

    return wrapper


def handle_ui_errors(func):
    """
    Decorator for UI components that displays errors in the Streamlit UI
    and returns a fallback value.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"UI Error: {str(e)}")
            traceback.print_exc()
            return None

    return wrapper


def safe_execute(func, *args, default=None, **kwargs):
    """
    Safely execute a function with given arguments, returning a default value if it fails.

    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        default: Default value to return if the function fails
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function or the default value if it fails
    """
    try:
        return func(*args, **kwargs)
    except Exception:
        traceback.print_exc()
        return default
