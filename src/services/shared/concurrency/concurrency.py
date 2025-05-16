#!/usr/bin/env python3
"""
Concurrency utility for the Program Synthesis System.

This module provides utilities for handling concurrent operations,
task parallelization, and asynchronous processing in the context
of program synthesis.
"""

import asyncio
import concurrent.futures
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import logging
import os
import time
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)


# Define type variables for generic typing
T = TypeVar("T")
R = TypeVar("R")


class TaskPriority(Enum):
    """Priority levels for concurrent tasks."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


@dataclass
class TaskInfo:
    """Information about a task in the task pool."""

    id: str
    name: str
    priority: TaskPriority
    created_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    status: str = "pending"
    result: Any = None
    error: Optional[Exception] = None


class TaskPool:
    """Manages a pool of concurrent tasks with priority scheduling."""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize the task pool.

        Args:
            max_workers: Maximum number of worker threads/processes (default: CPU count)
            use_processes: Whether to use processes instead of threads
        """
        self.logger = logging.getLogger("TaskPool")
        self.max_workers = max_workers or os.cpu_count() or 4
        self.use_processes = use_processes

        # Create executor
        if use_processes:
            self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers)
        else:
            self.executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="synthesis-worker"
            )

        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.futures: Dict[str, concurrent.futures.Future] = {}

        self.logger.info(
            f"Initialized task pool with {self.max_workers} workers (using {'processes' if use_processes else 'threads'})"
        )

    def submit(
        self,
        func: Callable[..., T],
        *args: Any,
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
        **kwargs: Any,
    ) -> str:
        """
        Submit a task to the pool.

        Args:
            func: Function to execute
            *args: Arguments for the function
            task_id: Optional task ID (auto-generated if not provided)
            task_name: Optional task name
            priority: Task priority
            **kwargs: Keyword arguments for the function

        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"task-{time.time()}-{len(self.tasks)}"

        # Use function name as task name if not provided
        if task_name is None:
            task_name = func.__name__

        # Create task info
        task_info = TaskInfo(id=task_id, name=task_name, priority=priority, created_at=time.time())

        # Store task info
        self.tasks[task_id] = task_info

        # Define callback for task completion
        def task_done_callback(future: concurrent.futures.Future) -> None:
            task_info.completed_at = time.time()
            try:
                task_info.result = future.result()
                task_info.status = "completed"
            except Exception as e:
                task_info.error = e
                task_info.status = "failed"
                self.logger.error(f"Task {task_id} ({task_name}) failed: {e}")

        # Submit task to executor based on priority
        # Unfortunately, ThreadPoolExecutor/ProcessPoolExecutor don't support priority natively,
        # so we're just tracking it in our task info
        self.logger.debug(f"Submitting task {task_id} ({task_name}) with priority {priority.name}")

        future = self.executor.submit(func, *args, **kwargs)
        future.add_done_callback(task_done_callback)

        # Store future for later access
        self.futures[task_id] = future

        return task_id

    def get_task_info(self, task_id: str) -> Optional[TaskInfo]:
        """
        Get information about a task.

        Args:
            task_id: Task ID

        Returns:
            Task information, or None if not found
        """
        return self.tasks.get(task_id)

    def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task, waiting if necessary.

        Args:
            task_id: Task ID
            timeout: Maximum time to wait (in seconds, None for no timeout)

        Returns:
            Task result

        Raises:
            KeyError: If task ID not found
            TimeoutError: If timeout expired
            Exception: If task failed
        """
        if task_id not in self.futures:
            raise KeyError(f"Task {task_id} not found")

        future = self.futures[task_id]

        # Get result, waiting if necessary
        result = future.result(timeout=timeout)

        return result

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if possible.

        Args:
            task_id: Task ID

        Returns:
            True if task was canceled, False otherwise
        """
        if task_id not in self.futures:
            return False

        future = self.futures[task_id]

        # Try to cancel the future
        canceled = future.cancel()

        if canceled:
            # Update task info
            task_info = self.tasks.get(task_id)
            if task_info:
                task_info.status = "canceled"
                task_info.completed_at = time.time()

        return canceled

    def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, str]:
        """
        Wait for multiple tasks to complete.

        Args:
            task_ids: List of task IDs
            timeout: Maximum time to wait (in seconds, None for no timeout)

        Returns:
            Dictionary mapping task IDs to status
        """
        # Get futures for all valid task IDs
        futures_to_wait = {self.futures[task_id]: task_id for task_id in task_ids if task_id in self.futures}

        if not futures_to_wait:
            return {}

        # Wait for futures to complete
        done, not_done = concurrent.futures.wait(futures_to_wait.keys(), timeout=timeout)

        # Collect results
        results = {}

        for future in done:
            task_id = futures_to_wait[future]
            task_info = self.tasks.get(task_id)
            if task_info:
                results[task_id] = task_info.status

        for future in not_done:
            task_id = futures_to_wait[future]
            results[task_id] = "pending"

        return results

    def shutdown(self, wait: bool = True, cancel_futures: bool = False) -> None:
        """
        Shut down the task pool.

        Args:
            wait: Whether to wait for pending tasks to complete
            cancel_futures: Whether to cancel pending futures
        """
        if cancel_futures:
            # Cancel all pending tasks
            for task_id, future in self.futures.items():
                if not future.done():
                    future.cancel()
                    task_info = self.tasks.get(task_id)
                    if task_info:
                        task_info.status = "canceled"

        # Shut down the executor
        self.executor.shutdown(wait=wait)

        self.logger.info(f"Shut down task pool (waited for completion: {wait})")


class AsyncTaskManager:
    """Manages asynchronous tasks with priority scheduling."""

    def __init__(self, max_concurrency: int = 10):
        """
        Initialize the async task manager.

        Args:
            max_concurrency: Maximum number of concurrent tasks
        """
        self.logger = logging.getLogger("AsyncTaskManager")
        self.max_concurrency = max_concurrency

        # Task tracking
        self.tasks: Dict[str, TaskInfo] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}

        # Semaphore for concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrency)

        # Task queues by priority
        self.task_queues: Dict[TaskPriority, asyncio.Queue] = {priority: asyncio.Queue() for priority in TaskPriority}

        # Flag to indicate shutdown
        self.shutting_down = False

        # Start task scheduler
        self.scheduler_task = asyncio.create_task(self._scheduler())

        self.logger.info(f"Initialized async task manager with max concurrency {max_concurrency}")

    async def _scheduler(self) -> None:
        """Task scheduler coroutine."""
        while not self.shutting_down:
            # Check each priority queue in order
            for priority in sorted(TaskPriority, key=lambda p: p.value, reverse=True):
                queue = self.task_queues[priority]

                if not queue.empty():
                    # Get task from queue
                    task_id, coro = await queue.get()

                    # Acquire semaphore (wait if too many tasks running)
                    await self.semaphore.acquire()

                    # Start task
                    asyncio_task = asyncio.create_task(self._run_task(task_id, coro))
                    self.running_tasks[task_id] = asyncio_task

                    # Mark queue task as done
                    queue.task_done()

                    # Found a task to run, break from priority loop
                    break

            # If no tasks were found, sleep briefly to avoid busy-waiting
            else:
                await asyncio.sleep(0.01)

    async def _run_task(self, task_id: str, coro: Coroutine) -> None:
        """
        Run a task and update its status.

        Args:
            task_id: Task ID
            coro: Coroutine to run
        """
        task_info = self.tasks.get(task_id)
        if not task_info:
            # Task info not found, release semaphore and exit
            self.semaphore.release()
            return

        # Update task status
        task_info.status = "running"
        task_info.started_at = time.time()

        try:
            # Run the task
            result = await coro

            # Update task info on success
            task_info.status = "completed"
            task_info.result = result
        except Exception as e:
            # Update task info on failure
            task_info.status = "failed"
            task_info.error = e
            self.logger.error(f"Task {task_id} ({task_info.name}) failed: {e}")
        finally:
            # Release semaphore
            self.semaphore.release()

            # Remove from running tasks
            if task_id in self.running_tasks:
                del self.running_tasks[task_id]

            # Update completion time
            task_info.completed_at = time.time()

    async def submit(
        self,
        coro: Coroutine[Any, Any, T],
        task_id: Optional[str] = None,
        task_name: Optional[str] = None,
        priority: TaskPriority = TaskPriority.MEDIUM,
    ) -> str:
        """
        Submit a coroutine to be executed.

        Args:
            coro: Coroutine to execute
            task_id: Optional task ID (auto-generated if not provided)
            task_name: Optional task name
            priority: Task priority

        Returns:
            Task ID
        """
        # Generate task ID if not provided
        if task_id is None:
            task_id = f"async-task-{time.time()}-{len(self.tasks)}"

        # Use default name if not provided
        if task_name is None:
            task_name = f"async-task-{task_id}"

        # Create task info
        task_info = TaskInfo(id=task_id, name=task_name, priority=priority, created_at=time.time())

        # Store task info
        self.tasks[task_id] = task_info

        # Add task to appropriate queue
        await self.task_queues[priority].put((task_id, coro))

        self.logger.debug(f"Submitted async task {task_id} ({task_name}) with priority {priority.name}")

        return task_id

    async def get_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get the result of a task, waiting if necessary.

        Args:
            task_id: Task ID
            timeout: Maximum time to wait (in seconds, None for no timeout)

        Returns:
            Task result

        Raises:
            KeyError: If task ID not found
            TimeoutError: If timeout expired
            Exception: If task failed
        """
        if task_id not in self.tasks:
            raise KeyError(f"Task {task_id} not found")

        task_info = self.tasks[task_id]

        # If task already completed, return result or raise error
        if task_info.status == "completed":
            return task_info.result
        elif task_info.status == "failed" and task_info.error is not None:
            raise task_info.error

        # Wait for task to complete
        start_time = time.time()
        while task_info.status not in ["completed", "failed", "canceled"]:
            # Check timeout
            if timeout is not None and time.time() - start_time > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")

            # Wait a bit
            await asyncio.sleep(0.05)

        # Task completed, return result or raise error
        if task_info.status == "completed":
            return task_info.result
        elif task_info.status == "failed" and task_info.error is not None:
            raise task_info.error
        else:
            raise Exception(f"Task {task_id} was canceled")

    async def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task if possible.

        Args:
            task_id: Task ID

        Returns:
            True if task was canceled, False otherwise
        """
        if task_id not in self.tasks:
            return False

        task_info = self.tasks[task_id]

        # Check if task is already completed
        if task_info.status in ["completed", "failed", "canceled"]:
            return False

        # If task is running, cancel the asyncio task
        if task_id in self.running_tasks:
            asyncio_task = self.running_tasks[task_id]
            asyncio_task.cancel()

            # Wait for task to be canceled
            try:
                await asyncio_task
            except asyncio.CancelledError:
                pass

            # Update task info
            task_info.status = "canceled"
            task_info.completed_at = time.time()

            # Remove from running tasks
            del self.running_tasks[task_id]

            return True

        # If task is still in queue, mark it as canceled
        # (It will still be processed by the scheduler, but will exit immediately)
        task_info.status = "canceled"
        task_info.completed_at = time.time()

        return True

    async def wait_for_tasks(self, task_ids: List[str], timeout: Optional[float] = None) -> Dict[str, str]:
        """
        Wait for multiple tasks to complete.

        Args:
            task_ids: List of task IDs
            timeout: Maximum time to wait (in seconds, None for no timeout)

        Returns:
            Dictionary mapping task IDs to status
        """
        # Filter valid task IDs
        valid_ids = [task_id for task_id in task_ids if task_id in self.tasks]

        if not valid_ids:
            return {}

        # Wait for tasks to complete
        results = {}
        pending_ids = set(valid_ids)

        # Set timeout
        end_time = None
        if timeout is not None:
            end_time = time.time() + timeout

        while pending_ids and (end_time is None or time.time() < end_time):
            # Check each task
            for task_id in list(pending_ids):
                task_info = self.tasks[task_id]

                # If task completed, add to results
                if task_info.status in ["completed", "failed", "canceled"]:
                    results[task_id] = task_info.status
                    pending_ids.remove(task_id)

            # If still waiting for tasks, sleep briefly
            if pending_ids:
                await asyncio.sleep(0.05)

        # Add any remaining tasks as "pending"
        for task_id in pending_ids:
            results[task_id] = "pending"

        return results

    async def shutdown(self, wait: bool = True, cancel_pending: bool = False) -> None:
        """
        Shut down the task manager.

        Args:
            wait: Whether to wait for pending tasks to complete
            cancel_pending: Whether to cancel pending tasks
        """
        self.shutting_down = True

        # Cancel the scheduler task
        self.scheduler_task.cancel()
        try:
            await self.scheduler_task
        except asyncio.CancelledError:
            pass

        if cancel_pending:
            # Cancel all running tasks
            for task_id, asyncio_task in list(self.running_tasks.items()):
                asyncio_task.cancel()
                task_info = self.tasks.get(task_id)
                if task_info:
                    task_info.status = "canceled"
                    task_info.completed_at = time.time()

        if wait:
            # Wait for all running tasks to complete
            if self.running_tasks:
                await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)

        self.logger.info(f"Shut down async task manager (waited for completion: {wait})")


class ParallelExecutor:
    """Executes functions in parallel with automatic parallelism detection."""

    def __init__(self, max_workers: Optional[int] = None, use_processes: bool = False):
        """
        Initialize the parallel executor.

        Args:
            max_workers: Maximum number of worker threads/processes (default: CPU count)
            use_processes: Whether to use processes instead of threads
        """
        self.logger = logging.getLogger("ParallelExecutor")
        self.max_workers = max_workers or os.cpu_count() or 4
        self.use_processes = use_processes

        # Task pool for parallel execution
        self.task_pool = TaskPool(max_workers=max_workers, use_processes=use_processes)

        self.logger.info(f"Initialized parallel executor with {self.max_workers} workers")

    def map(self, func: Callable[[T], R], items: List[T], chunk_size: Optional[int] = None) -> List[R]:
        """
        Apply a function to each item in a list in parallel.

        Args:
            func: Function to apply
            items: List of items
            chunk_size: Number of items to process in each task (auto-determined if None)

        Returns:
            List of results in the same order as items
        """
        if not items:
            return []

        # Determine chunk size if not provided
        if chunk_size is None:
            chunk_size = max(1, min(100, len(items) // (self.max_workers * 2)))

        # Create chunks
        chunks = [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

        # Define function to process a chunk
        def process_chunk(chunk: List[T]) -> List[R]:
            return [func(item) for item in chunk]

        # Submit tasks for each chunk
        task_ids = [
            self.task_pool.submit(process_chunk, chunk, task_name=f"map-chunk-{i}") for i, chunk in enumerate(chunks)
        ]

        # Collect results
        results: List[List[R]] = []
        for task_id in task_ids:
            chunk_results = self.task_pool.get_result(task_id)
            results.append(chunk_results)

        # Flatten results
        return [result for chunk_result in results for result in chunk_result]

    def execute_all(self, tasks: List[Tuple[Callable[..., R], List[Any], Dict[str, Any]]]) -> List[R]:
        """
        Execute a list of tasks in parallel.

        Args:
            tasks: List of (function, args, kwargs) tuples

        Returns:
            List of results in the same order as tasks
        """
        if not tasks:
            return []

        # Submit tasks
        task_ids = []
        for i, (func, args, kwargs) in enumerate(tasks):
            task_id = self.task_pool.submit(func, *args, task_name=f"execute-all-{i}", **kwargs)
            task_ids.append(task_id)

        # Collect results
        results = []
        for task_id in task_ids:
            try:
                result = self.task_pool.get_result(task_id)
                results.append(result)
            except Exception as e:
                # Include exception as result
                results.append(e)

        return results

    def shutdown(self) -> None:
        """Shut down the executor."""
        self.task_pool.shutdown()


@contextmanager
def parallel_context(max_workers: Optional[int] = None, use_processes: bool = False):
    """
    Context manager for parallel execution.

    Args:
        max_workers: Maximum number of worker threads/processes
        use_processes: Whether to use processes instead of threads

    Yields:
        ParallelExecutor instance
    """
    executor = ParallelExecutor(max_workers=max_workers, use_processes=use_processes)
    try:
        yield executor
    finally:
        executor.shutdown()


def run_parallel(
    func: Callable[[T], R],
    items: List[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False,
    chunk_size: Optional[int] = None,
) -> List[R]:
    """
    Run a function on a list of items in parallel.

    Args:
        func: Function to apply to each item
        items: List of items
        max_workers: Maximum number of worker threads/processes
        use_processes: Whether to use processes instead of threads
        chunk_size: Number of items to process in each task

    Returns:
        List of results in the same order as items
    """
    with parallel_context(max_workers=max_workers, use_processes=use_processes) as executor:
        return executor.map(func, items, chunk_size=chunk_size)


async def gather_with_concurrency(coros: List[Coroutine[Any, Any, T]], limit: int = 10) -> List[T]:
    """
    Run coroutines with a concurrency limit.

    Args:
        coros: List of coroutines
        limit: Maximum number of coroutines to run at once

    Returns:
        List of results in the same order as coroutines
    """
    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(limit)

    # Define wrapper for coroutines
    async def semaphore_coro(coro: Coroutine[Any, Any, T]) -> T:
        async with semaphore:
            return await coro

    # Run coroutines with semaphore
    return await asyncio.gather(*(semaphore_coro(coro) for coro in coros))
