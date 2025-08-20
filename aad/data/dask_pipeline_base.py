import multiprocessing as mp
from dask.distributed import Client
from dask.diagnostics.progress import ProgressBar
from dask.base import compute
from typing import List, Any, Optional


class DaskPipelineBase:
    def run_distributed_tasks(
        self,
        tasks: List[Any],
        client=None,
        logger: Optional[Any] = None,
    ) -> List[Any]:
        """
        Run Dask tasks using an existing Dask client with optional progress bar and logging.

        Args:
            tasks: List of Dask delayed objects to compute.
            client: Existing Dask distributed.Client instance.
            logger: Optional logger for logging cluster info.

        Returns:
            List of computed task results.
        """
        if client is None:
            raise ValueError("A Dask client must be provided to run_distributed_tasks.")
        if logger and hasattr(client, "dashboard_link"):
            logger.log_step(f"Using existing Dask cluster: {client.dashboard_link}")
        with ProgressBar():
            results = list(compute(*tasks, scheduler=client.get))
        return results

    def run_distributed_map(
        self,
        func,
        iterable_args,
        client=None,
        logger: Optional[Any] = None,
        aggregate_fn=None,
        pure: bool = False,
    ):
        """
        Generic distributed map: applies func to each item in iterable_args in parallel, then aggregates results.

        Args:
            func: Function to apply (should be picklable).
            iterable_args: Iterable of argument tuples or single arguments.
            client: Dask client.
            logger: Optional logger.
            aggregate_fn: Optional aggregation function to combine results.
            pure: Whether to treat delayed as pure.

        Returns:
            Aggregated result if aggregate_fn is provided, else list of results.
        """
        from dask.delayed import delayed

        tasks = [delayed(func, pure=pure)(*args) if isinstance(args, (tuple, list)) else delayed(func, pure=pure)(args) for args in iterable_args]
        results = self.run_distributed_tasks(tasks, client=client, logger=logger)
        if aggregate_fn:
            return aggregate_fn(results)
        return results
