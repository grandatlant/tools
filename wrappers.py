"""Decorators and other tools to enhance function usage."""

__version__ = '1.0.4'
__copyright__ = 'Copyright (C) 2025 grandatlant'

__all__ = [
    'no_op',
    'no_op_func',
    'retry',
    'log_perf_counter',
    'wrap_with_calls',
    'wrap_with',
    'call_before',
    'call_after',
]


import time
import logging
import functools
import itertools

from typing import (
    Any,
    Type,
    Union,
    Optional,
    Tuple,
    Callable,
)


def no_op_func(*args, **kwargs):
    """Wrapper for doing completely nothing."""
    return None  # TODO: Think about return value here


def no_op(wrapped: Callable):
    """Decorator to make wrapped func do nothing."""
    return functools.update_wrapper(no_op_func, wrapped)


def retry(
    count: Union[int, Callable] = 3,
    delay: Optional[float] = None,
    delay_func: Callable[[float], Any] = time.sleep,
    exceptions: Union[
        Type[BaseException], Tuple[Type[BaseException], ...]
    ] = Exception,
    logger: Optional[logging.Logger] = None,
    log_args: bool = False,
    default: Any = Ellipsis,  # Ellipsis here because 'None' is OK 'default'
) -> Callable:
    """Decorator for performing retry logic on 'exceptions' occured.

    Parameters:
        count (int | Callable):
            maximum tries to call decorated function,
            or Callable to allow use as @retry with no perenteses.
            Default value: 3 (yes, just 3, don't ask me why)
        delay (float | None):
            delay after retries before next call try.
            delay_func will be called with this parameter value.
            Default value: None
        delay_func (Callable[[float], Any]):
            function to use as delay handler with 'delay' param if not None.
            Can be used as callback for failed try.
            Default value: time.sleep
        exceptions (BaseException | tuple):
            Expression used in 'except' clause for 'try' block.
            Exception class or tuple with exception classes to catch for retry.
            Default value: Exception
                catch all non-exit exceptions in default mode.
        logger (logging.Logger | None):
            object to use 'exception' method for logging exceptions occured.
            'exception' method called with parameters
                msg: str = log message, including func module, name,
                retry iteration number and exception repr.
            Default value: None
                no exception logging performed in this case.
        log_args (bool):
            Flag to log also *args and **kwargs for decorated func call.
            Default value: False
        default (Any):
            Value to return in case of all ('count') retry attempts failed.
            Default value: Ellipsis
                last exception will be reraised in this case
                with 'raise' statement.
    """

    decorated = None
    if callable(count) and delay is None:
        # called as @retry. reset count to default and save func
        decorated, count = count, 3
    elif 0 >= count:  # type: ignore
        # No need 'count' instancecheck, just check if
        # this comparison types allowed for decorator work.
        count = 1  # atleast 1 try will be performed anyway. Just explicit set
        # Note: use @no_op decorator to make 0 tries.

    def decorator(func):
        _func_trace = '%s.%s' % (
            func.__module__,
            func.__name__,
        )

        def wrapper(*func_args, **func_kwargs):
            _func_call_trace = (
                '{func}(*{args}, **{kwds})'.format(
                    func=_func_trace,
                    args=func_args,
                    kwds=func_kwargs,
                )
                if log_args
                else _func_trace
            )

            for iteration in itertools.count(1, 1):
                try:
                    return func(*func_args, **func_kwargs)
                except exceptions as exc:
                    if logger is not None:
                        logger.exception(
                            'Exception in %s call, @retry iteration #%s: %r.'
                            % (
                                _func_call_trace,
                                iteration,
                                exc,
                            )
                        )
                    # Stop trying and break conditions
                    if iteration >= count:
                        if default is Ellipsis:
                            raise
                        else:
                            return default
                if delay is not None:
                    delay_func(delay)

        return functools.update_wrapper(wrapper, func)

    if decorated is not None:  # called as @retry
        return decorator(decorated)
    return decorator  # called as @retry(...)


def log_perf_counter(
    param: Optional[Callable] = None,
    /,
    *,
    perf_counter: Callable[[], Any] = time.perf_counter,
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    log_call: bool = False,
    log_args: bool = False,
) -> Callable:
    r"""Decorator for performance measurement purposes,
    using perf_counter parameter as callable (default time.perf_counter_ns).

    Parameters:
        param (Callable | None):
            function to be decorated when using @log_perf_counter
            with no perenteses with all default parameters.
            Default value: None
        perf_counter (Callable):
            function with no arguments, returning value
            with '__sub__' method implemented to determine time-delta,
            waisted for decorated function call.
            Default value: time.perf_counter
        logger (logging.Logger):
            object to use 'log' method for logging decorator activity.
            'log' method called with parameters
                level: int = level for logging
                msg: str = log message
            Default value: None
                logging.getLogger() called in this case
                for each decorated function in format
                '__module__.__name__'
        level (int):
            level for new log messages.
            Default value: logging.DEBUG
        log_call (bool):
            boolean flag for logging additional information about
            perf_counter when function call starts and ends
            Default value: False
        log_args (bool):
            boolean flag for logging additional information about
            args and kwargs passed to decorated function.
            WARNING! Can be resourse-expensive while getting args repr
            Default value: False

    Returns:
        Callable: The decorator or wrapper function depends on param.
    """

    def decorator(func):
        _func_trace = '%s.%s' % (
            func.__module__,
            func.__name__,
        )
        _logger = logging.getLogger(_func_trace) if logger is None else logger

        def wrapper(*func_args, **func_kwds):
            _func_call_trace = (
                '{func}(*{args}, **{kwds})'.format(
                    func=_func_trace,
                    args=func_args,
                    kwds=func_kwds,
                )
                if log_args
                else _func_trace
            )

            if log_call:
                _logger.log(level, 'Call %s start.' % _func_call_trace)

            pc_before = perf_counter()
            func_return = func(*func_args, **func_kwds)
            pc_after = perf_counter()
            pc_delta = pc_after - pc_before

            if log_call:
                _logger.log(
                    level,
                    'Call %s finish. '
                    'Start: %s. '
                    'Finish: %s. '
                    'Delta: %s.'
                    % (
                        _func_call_trace,
                        pc_before,
                        pc_after,
                        pc_delta,
                    ),
                )
            else:
                _logger.log(
                    level,
                    '%s call time: %s'
                    % (
                        _func_call_trace,
                        pc_delta,
                    ),
                )

            return func_return

        return functools.update_wrapper(wrapper, func)

    if param is None:  # called as @log_perf_counter(...)
        return decorator
    return decorator(param)  # called as @log_perf_counter


def wrap_with_calls(
    func: Optional[Callable] = None,
    *func_args,
    first_call: Optional[Callable] = None,
    after_call: Optional[Callable] = None,
    args: Optional[tuple] = None,
    kwds: Optional[dict] = None,
    return_filter_func: Optional[Callable[[Any], bool]] = None,
    reduce_result_func: Optional[Callable[[Any, Any], Any]] = None,
) -> Callable:
    r"""Decorator to execute specified functions
    before and after the decorated function.

    Parameters:
        func (Callable):
            a single function to call before AND after
            the decorated function called.
            if wrap_with_calls used without (), transforms func
            into decorator, can be applied to other functions
        first_call (Callable or Iterable[Callable]):
            Function(s) to call before the decorated function.
        after_call (Callable or Iterable[Callable]):
            Function(s) to call after the decorated function.
        args (Tuple):
            Positional arguments to pass to all callables.
        kwds (Dict):
            Keyword arguments to pass to all callables.
        return_filter_func (Callable):
            Filter function to apply to return values of callables.
            If return_filter_func(returned_value) True -> return returned_value
            and stop next processing.
        reduce_result_func (Callable):
            Function to reduce results of all calls into one value.

    Returns:
        Callable: The decorated function.
    """

    def list_of_callables(callables):
        """Ensure the input is an iterable of callables."""
        if callable(callables):
            return list((callables,))
        elif hasattr(callables, '__iter__'):
            return list(callables)
        else:
            # I dont want exceptions here for dynamic use
            return list()
            # raise ValueError(f'Parameter "{callables}" '
            #                 'must be a callable '
            #                 'or an iterable of callables.')

    _args = args or tuple()
    _kwds = kwds or dict()

    def decorator(decorated_func):
        def decorated_func_wrapper(
            *decorated_func_args, **decorated_func_kwds
        ):
            results = list()

            # first calls
            for item in list_of_callables(first_call):
                if callable(item):
                    cur_result = item(*_args, **_kwds)
                    if callable(return_filter_func) and return_filter_func(
                        cur_result
                    ):
                        return cur_result
                    results.append(cur_result)

            # func before decorated_func
            if callable(func):
                func_result = func(*func_args, *_args, **_kwds)
                if callable(return_filter_func) and return_filter_func(
                    func_result
                ):
                    return func_result
                results.append(func_result)

            # !!! decorated_func call !!!
            decorated_func_result = decorated_func(
                *decorated_func_args, **decorated_func_kwds
            )
            ##TODO: Think about next 2 commented lines...
            # if return_filter_func(decorated_func_result):
            #    return decorated_func_result
            results.append(decorated_func_result)

            # func after decorated_func
            if callable(func):
                func_result = func(*func_args, *_args, **_kwds)
                if callable(return_filter_func) and return_filter_func(
                    func_result
                ):
                    return func_result
                results.append(func_result)

            # after calls
            for item in list_of_callables(after_call):
                if callable(item):
                    cur_result = item(*_args, **_kwds)
                    if callable(return_filter_func) and return_filter_func(
                        cur_result
                    ):
                        return cur_result
                    results.append(cur_result)

            # reduce results if specified
            if callable(reduce_result_func):
                return functools.reduce(reduce_result_func, results)

            # general result
            return decorated_func_result

        return functools.update_wrapper(decorated_func_wrapper, decorated_func)

    return decorator


def wrap_with(
    func_before: Optional[Callable] = None,
    func_after: Optional[Callable] = None,
    *func_args,
    args: Optional[tuple] = None,
    kwds: Optional[dict] = None,
    return_filter_func: Optional[Callable[[Any], bool]] = None,
    reduce_result_func: Optional[Callable[[Any, Any], Any]] = None,
) -> Callable:
    r"""Wrapper for 'wrap_with_calls' decorator
    for call 'func_before' before decorated function
    and then call 'func_after' after decorated function execution.
    Next positional decorator arguments passed to both callables.
    """
    return wrap_with_calls(
        first_call=func_before,
        after_call=func_after,
        args=(*func_args, *(args or tuple())),
        kwds=kwds,
        return_filter_func=return_filter_func,
        reduce_result_func=reduce_result_func,
    )


def call_before(
    func: Callable,
    *func_args,
    args: Optional[tuple] = None,
    kwds: Optional[dict] = None,
    return_filter_func: Optional[Callable[[Any], bool]] = None,
    reduce_result_func: Optional[Callable[[Any, Any], Any]] = None,
) -> Callable:
    r"""Wrapper for 'wrap_with_calls' decorator
    for call 'func' before decorated function execution.
    Next positional decorator arguments passed to callable 'func'.
    """
    return wrap_with_calls(
        first_call=func,
        args=(*func_args, *(args or tuple())),
        kwds=kwds,
        return_filter_func=return_filter_func,
        reduce_result_func=reduce_result_func,
    )


def call_after(
    func: Callable,
    *func_args,
    args: Optional[tuple] = None,
    kwds: Optional[dict] = None,
    return_filter_func: Optional[Callable[[Any], bool]] = None,
    reduce_result_func: Optional[Callable[[Any, Any], Any]] = None,
) -> Callable:
    r"""Wrapper for 'wrap_with_calls' decorator
    for call 'func' after decorated function execution.
    Next positional decorator arguments passed to callable 'func'.
    """
    return wrap_with_calls(
        after_call=func,
        args=(*func_args, *(args or tuple())),
        kwds=kwds,
        return_filter_func=return_filter_func,
        reduce_result_func=reduce_result_func,
    )
