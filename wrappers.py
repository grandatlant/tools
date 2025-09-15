"""Decorators and other tools to enhance function usage.
"""

__version__ = '1.0.2'
__copyright__ = 'Copyright (C) 2025 grandatlant'

__all__ = [
    'log_perf_counter',
    'wrap_with_calls',
    'wrap_with',
    'call_before',
    'call_after',
]


import time
import logging
import functools

from typing import (
    Any,
    Optional,
    Callable,
)


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
        _func_trace: str = '%s.%s' % (
            func.__module__,
            func.__name__,
        )
        _logger: logging.Logger = logger or logging.getLogger(_func_trace)

        @functools.wraps(func)
        def wrapper(*func_args, **func_kwds):
            if log_args:
                _func_call_trace: str = '{func}(*{args}, **{kwds})'.format(
                    func = _func_trace,
                    args = func_args,
                    kwds = func_kwds,
                )
            else:
                _func_call_trace: str = _func_trace
            
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
                    'Delta: %s.' % (
                        _func_call_trace,
                        pc_before,
                        pc_after,
                        pc_delta,
                    )
                )
            else:
                _logger.log(level, '%s call time: %s' % (
                    _func_call_trace,
                    pc_delta,
                ))
            
            return func_return
        
        return wrapper

    if param is None:
        # called as @log_perf_counter(...)
        return decorator
    # called as @log_perf_counter
    return decorator(param)


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
            #raise ValueError(f'Parameter "{callables}" '
            #                 'must be a callable '
            #                 'or an iterable of callables.')
    
    _args = args or tuple()
    _kwds = kwds or dict()
    
    @functools.wraps(func) # type: ignore
    def decorator(decorated_func):
        @functools.wraps(decorated_func)
        def decorated_func_wrapper(
            *decorated_func_args,
            **decorated_func_kwds
        ):
            results = list()

            # first calls
            for item in list_of_callables(first_call):
                if callable(item):
                    cur_result = item(*_args, **_kwds)
                    if (callable(return_filter_func)
                        and return_filter_func(cur_result)):
                        return cur_result
                    results.append(cur_result)

            # func before decorated_func
            if callable(func):
                func_result = func(*func_args, *_args, **_kwds)
                if (callable(return_filter_func)
                    and return_filter_func(func_result)):
                    return func_result
                results.append(func_result)

            # !!! decorated_func call !!!
            decorated_func_result = decorated_func(
                *decorated_func_args,
                **decorated_func_kwds
            )
            ##TODO: Think about next 2 commented lines...
            #if return_filter_func(decorated_func_result):
            #    return decorated_func_result
            results.append(decorated_func_result)

            # func after decorated_func
            if callable(func):
                func_result = func(*func_args, *_args, **_kwds)
                if (callable(return_filter_func)
                    and return_filter_func(func_result)):
                    return func_result
                results.append(func_result)

            # after calls
            for item in list_of_callables(after_call):
                if callable(item):
                    cur_result = item(*_args, **_kwds)
                    if (callable(return_filter_func)
                        and return_filter_func(cur_result)):
                        return cur_result
                    results.append(cur_result)
            
            # reduce results if specified
            if callable(reduce_result_func):
                return functools.reduce(reduce_result_func, results)
            
            # general result
            return decorated_func_result
        return decorated_func_wrapper
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
