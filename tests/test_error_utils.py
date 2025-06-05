import lightlygpt.utils.error_utils as error_utils


def test_handle_tool_errors_decorator():
    @error_utils.handle_tool_errors
    def func_ok():
        return "ok"

    @error_utils.handle_tool_errors
    def func_fail():
        raise ValueError("fail")

    assert func_ok() == "ok"
    result = func_fail()
    assert "fail" in result or "error" in result.lower()
