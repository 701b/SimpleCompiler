"""
Stack
-----
Team Name : 떡볶이 국물이 옷에 팀
Student Name : 문태의, 성아영
"""

class Stack:
    """자료구조 Stack의 기초적인 기능을 제공하는 클래스"""

    def __init__(self):
        self._stack = []
        self._current = 0

    def push(self, data) -> None:
        """스택에 데이터를 추가한다.

        Args:
            data: 스택에 추가할 데이터
        """
        self._stack.append(data)

    def pop(self):
        """스택의 가장 위에 있는 데이터를 꺼낸다.

        Returns: 스택의 가장 위에 있는 데이터
        """
        return self._stack.pop()

    def get(self):
        """스택의 가장 위에 있는 데이터를 반환하지만 데이터를 꺼내지는 않는다.

        Returns: 스택의 가장 위에 있는 데이터
        """
        return self._stack[-1]

    def is_empty(self) -> bool:
        """스택이 비어있는지 확인한다.

        Returns: 스택이 비어있다면 True, 비어있지 않다면 False
        """
        return len(self._stack) == 0

    def __str__(self):
        text = ""

        for i in range(-1, -(len(self._stack) + 1), -1):
            text += self._stack[i] + " "

        return text

    def __iter__(self):
        return self

    def __next__(self):
        if self._current < len(self._stack):
            self._current += 1
            return self._stack[-self._current]
        else:
            self._current = 0
            raise StopIteration


class Queue:
    """자료구조 Queue의 기초적인 기능을 제공하는 클래스"""

    def __init__(self):
        self._queue = []
        self._current = 0

    def enqueue(self, data) -> None:
        """큐에 데이터를 추가한다.

        Args:
            data: 큐에 추가할 데이터
        """
        self._queue.append(data)

    def dequeue(self):
        """큐에서 가장 앞에 있는 데이터를 꺼낸다.

        Returns: 큐의 가장 앞에 있는 데이터
        """
        data = self._queue[0]
        del self._queue[0]
        return data

    def get(self):
        """큐에서 가장 앞에 있는 데이터를 반환하지만 꺼내진 않는다.

        Returns: 큐에서 가장 앞에 있는 데이터
        """
        return self._queue[0]

    def is_empty(self) -> bool:
        """큐가 비어있는지 확인한다.

        Returns: 큐가 비어있다면 True, 비어있지 않다면 False
        """
        return len(self._queue) == 0

    def __str__(self):
        text = ""

        for data in self._queue:
            text += f"{data} "

        return text

    def __iter__(self):
        return self

    def __next__(self):
        if self._current < len(self._queue):
            self._current += 1
            return self._queue[self._current - 1]
        else:
            self._current = 0
            raise StopIteration
