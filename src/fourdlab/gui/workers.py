"""Small Qt worker helper for long-running GUI tasks."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot


TaskFunction = Callable[[Callable[[int, int], None], Callable[[], bool]], object]


class Worker(QObject):
    """Run one function in a QThread and report progress safely to the GUI."""

    progress = pyqtSignal(int, int)
    finished = pyqtSignal(object)
    failed = pyqtSignal(object)
    cancelled = pyqtSignal(str)

    def __init__(
        self,
        function: TaskFunction,
        *,
        cancelled_exception: type[BaseException] | tuple[type[BaseException], ...] = (),
    ) -> None:
        super().__init__()
        self._function = function
        self._cancelled_exception = cancelled_exception
        self._stop_requested = False

    def run(self) -> None:
        """Execute the task body."""

        try:
            result = self._function(self.progress.emit, lambda: self._stop_requested)
        except self._cancelled_exception as exc:
            self.cancelled.emit(str(exc))
        except Exception as exc:
            self.failed.emit(exc)
        else:
            self.finished.emit(result)

    def request_stop(self) -> None:
        """Ask the task to stop at its next cooperative cancellation check."""

        self._stop_requested = True


class CallbackRelay(QObject):
    """Own GUI-thread callbacks for a background task."""

    def __init__(
        self,
        *,
        on_progress: Callable[[int, int], None] | None = None,
        on_finished: Callable[[object], None] | None = None,
        on_failed: Callable[[BaseException], None] | None = None,
        on_cancelled: Callable[[str], None] | None = None,
        on_done: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self._on_progress = on_progress
        self._on_finished = on_finished
        self._on_failed = on_failed
        self._on_cancelled = on_cancelled
        self._on_done = on_done

    @pyqtSlot(int, int)
    def progress(self, done: int, total: int) -> None:
        if self._on_progress is not None:
            self._on_progress(done, total)

    @pyqtSlot(object)
    def finished(self, result: object) -> None:
        if self._on_finished is not None:
            self._on_finished(result)
        self.done()

    @pyqtSlot(object)
    def failed(self, exc: BaseException) -> None:
        if self._on_failed is not None:
            self._on_failed(exc)
        self.done()

    @pyqtSlot(str)
    def cancelled(self, message: str) -> None:
        if self._on_cancelled is not None:
            self._on_cancelled(message)
        self.done()

    def done(self) -> None:
        if self._on_done is not None:
            self._on_done()


@dataclass
class RunningTask:
    """Owns a worker thread until it finishes."""

    thread: QThread
    worker: Worker
    relay: CallbackRelay

    def request_stop(self) -> None:
        self.worker.request_stop()


def start_background_task(
    parent: QObject,
    function: TaskFunction,
    *,
    cancelled_exception: type[BaseException] | tuple[type[BaseException], ...] = (),
    on_progress: Callable[[int, int], None] | None = None,
    on_finished: Callable[[object], None] | None = None,
    on_failed: Callable[[BaseException], None] | None = None,
    on_cancelled: Callable[[str], None] | None = None,
    on_done: Callable[[], None] | None = None,
) -> RunningTask:
    """Start a cooperative background task and wire up lifecycle cleanup."""

    thread = QThread(parent)
    worker = Worker(function, cancelled_exception=cancelled_exception)
    relay = CallbackRelay(
        on_progress=on_progress,
        on_finished=on_finished,
        on_failed=on_failed,
        on_cancelled=on_cancelled,
        on_done=on_done,
    )
    relay.setParent(parent)
    worker.moveToThread(thread)

    worker.progress.connect(relay.progress)
    worker.finished.connect(relay.finished)
    worker.failed.connect(relay.failed)
    worker.cancelled.connect(relay.cancelled)
    worker.finished.connect(lambda _result: thread.quit())
    worker.failed.connect(lambda _exc: thread.quit())
    worker.cancelled.connect(lambda _message: thread.quit())
    thread.started.connect(worker.run)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(relay.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return RunningTask(thread=thread, worker=worker, relay=relay)
