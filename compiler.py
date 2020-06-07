"""
Compiler
--------
Team Name : 떡볶이 국물이 옷에 팀
Student Name : 문태의, 성아영
"""

import re
from filemanager import FileReader
from datastructure import Stack, Queue


TAG = "Compiler"


def log(content: str) -> None:
    """터미널에 로그를 찍기 위한 함수"""
    print(f"{TAG} >> {content}")


class Compiler:
    """컴파일러 클래스

    컴파일을 수행하기 위한 모든 클래스를 제어한다.

    compile 메서드를 통해 소스 파일의 코드를 타겟 코드로 컴파일한다.
    """

    def __init__(self):
        pass

    def compile(self, file_path: str) -> None:
        """컴파일을 시작하는 메서드

        Args:
            file_path: 컴파일할 소스 코드가 작성된 파일의 주소
        """
        # file_path로 부터 소스 코드를 읽어온다.
        log(f"다음의 경로에서 소스 코드를 불러옵니다 : {file_path}")
        file_reader = FileReader(file_path)
        source_code = file_reader.read_all()
        log(f"읽어온 소스 코드::\n{source_code}\n")

        scanner = Scanner(source_code)

        try:
            scanner.scan(file_path)
        except FileNotFoundError:
            log(f"다음의 주소에서 파일을 읽지 못했습니다 : {file_path}")
            return
        except InvalidTokenError as e:
            print(e)
            return

        parser = Parser(scanner.get_token_queue(), source_code)

        try:
            parser.parse()
        except NotMatchedBraceError as e:
            print(e)
            return


class Scanner:
    """컴파일러의 어휘 분석 단계를 수행하는 클래스

    메서드 analysis가 어휘 분석의 모든 단계를 수행하도록 한다.
    """

    TAG = "Scanner"

    REGULAR_EXPRESSION_OF_TOKEN = {'(': '\(',
                                   ')': '\)',
                                   '{': '\{',
                                   '}': '\}',
                                   ',': ',',
                                   ';': ';',
                                   '==': '==',
                                   '=': '=',
                                   '>': '>',
                                   '+': '\+',
                                   '*': '\*',
                                   'int': 'int',
                                   'char': 'char',
                                   'IF': 'IF',
                                   'ELSE': 'ELSE',
                                   'THEN': 'THEN',
                                   'WHILE': 'WHILE',
                                   'RETURN': 'RETURN',
                                   'word': '([a-z]|[A-Z])+',
                                   'num': '[0-9]+'}

    def __init__(self, source_code: str):
        self._source_code = source_code
        self._tokens = []

    def scan(self, file_path: str) -> None:
        """어휘 분석을 시작하는 메서드

        소스 코드로부터 토큰을 추출한다.
        토큰은 Token클래스에 저장되어 token 리스트에 담긴다.

        Args:
            file_path: 분석할 소스 코드가 작성된 파일의 주소

        Raises:
            FileNotFoundError: 인자로 받은 file_path로부터 파일을 읽지 못한 경우 발생한다.
            InvalidCodeError: 소스 코드에 알 수 없는 기호가 있을 때 발생한다.
        """

        # 소스 코드로부터 토큰을 추출한다.
        log("준비된 코드를 토큰으로 분리합니다.")
        source_code_temp = self._source_code
        line_number = 1

        while source_code_temp != "":
            is_match = False

            for key, value in Scanner.REGULAR_EXPRESSION_OF_TOKEN.items():
                pattern = re.compile(value)
                result = pattern.match(source_code_temp)

                if result is not None:
                    log(f"{key}:{result.group()}")
                    token = Token(key, result.group(), line_number)
                    self._tokens.append(token)
                    source_code_temp = source_code_temp.replace(result.group(), "", 1)
                    is_match = True
                    break

            if is_match is False:
                # 미리 정해진 정규 표현식에 패턴 매칭이 되지 않았다면, 공백과 줄바꿈인지 확인한다.
                pattern = re.compile("\s|\n")
                result = pattern.match(source_code_temp)

                if result is not None:
                    source_code_temp = source_code_temp.replace(result.group(), "", 1)
                    if result.group() == '\n':
                        # 줄바꿈이라면 line_number에 1 추가
                        line_number += 1
                if result is None:
                    # 만약 공백과 문자열이 아니라면 에러를 발생시킨다.
                    pattern = re.compile("[^\s\n]*")
                    invalid_token = pattern.match(source_code_temp)

                    raise InvalidTokenError(self._source_code, invalid_token.group(), line_number)

        log(f"토큰 목록::\n{self._tokens}\n")

    def get_token_queue(self) -> Queue:
        """토큰이 담긴 Queue를 반환한다.

        Returns: 토큰이 왼쪽부터 순서대로 담긴 Queue
        """
        queue = Queue()

        for token in self._tokens:
            queue.enqueue(token)

        return queue

    @property
    def source_code(self) -> str:
        return self._source_code


class Parser:
    """컴파일러의 구문 분석 단계를 담당하는 클래스

    파싱 테이블을 토대로 토큰들을 분석한다.
    """

    START_SYMBOL = "prog"

    PARSING_TABLE = {"prog": {"word": ["word", "(", ")", "block"],
                              "$": ["e"]},
                     "decls": {"word": ["e"],
                               "}": ["e"],
                               "IF": ["e"],
                               "WHILE": ["e"],
                               "RETURN": ["e"],
                               "int": ["decl", "decls'"],
                               "char": ["decl", "decls'"]},
                     "decl": {"int": ["vtype", "words", ";"],
                              "char": ["vtype", "words", ";"]},
                     "words": {"word": ["word", "words'"]},
                     "words'": {";": ["e"],
                                ",": [",", "word", "words'"]},
                     "vtype": {"word": ["e"],
                               "int": ["int"],
                               "char": ["char"]},
                     "block": {"word": ["e"],
                               "{": ["{", "decls", "slist", "}"],
                               "}": ["e"],
                               "IF": ["e"],
                               "WHILE": ["e"],
                               "RETURN": ["e"],
                               "ELSE": ["e"],
                               "$": ["e"]},
                     "slist": {"word": ["stat", "slist"],
                               "}": ["e"],
                               "IF": ["stat", "slist"],
                               "WHILE": ["stat", "slist"],
                               "RETURN": ["stat", "slist"]},
                     "stat": {"word": ["word", "=", "expr", ";"],
                              "IF": ["IF", "cond", "THEN", "block", "ELSE", "block"],
                              "WHILE": ["WHILE", "cond", "block"],
                              "RETURN": ["RETURN", "expr", ";"]},
                     "cond": {"word": ["expr", "expr'"],
                              "num": ["expr", "expr'"]},
                     "expr'": {"==": ["==", "expr"],
                               ">": [">", "expr"]},
                     "expr": {"word": ["term", "term'"],
                              "num": ["term", "term'"]},
                     "term'": {"+": ["+", "term"],
                               "{": ["e"],
                               ";": ["e"],
                               "==": ["e"],
                               ">": ["e"],
                               "THEN": ["e"]},
                     "term": {"word": ["fact", "fact'"],
                              "num": ["fact", "fact'"]},
                     "fact'": {"+": ["e"],
                               "*": ["*", "fact"],
                               "{": ["e"],
                               ";": ["e"],
                               "==": ["e"],
                               ">": ["e"],
                               "THEN": ["e"]},
                     "fact": {"word": ["word"],
                              "num": ["num"]}}

    def __init__(self, token_queue: Queue, source_code: str):
        """
        Args:
            token_queue: parse하는데 사용할 token이 담긴 queue
            source_code: 전체 소스코드
        """
        self._token_queue = token_queue
        self._source_code = source_code

    def parse(self) -> None:
        """Scanner로부터 받아온 Token들을 parse하는 메서드

        최초 기호와 마지막 기호가 담긴 stack과 Token들이 담긴 queue를 이용하여
        PARSING_TABLE에 따라 parse한다.
        LL(1) parser의 원리를 사용한다.
        """
        used_token_stack = Stack()
        stack = Stack()

        stack.push("$")
        stack.push(Parser.START_SYMBOL)

        self._token_queue.enqueue("$")

        log(f"queue : {self._token_queue}")
        log(f"stack : {stack}")
        log(f"work  : start\n")

        try:
            while not stack.is_empty():
                if self._token_queue.get().token == stack.get():
                    # 기호가 같을 때 pop
                    used_token_stack.push(self._token_queue.dequeue())
                    stack.pop()
                    log(f"queue : {self._token_queue}")
                    log(f"stack : {stack}")
                    log(f"work  : pop\n")
                else:
                    # 기호가 다를 땐 확장
                    symbol = stack.pop()
                    next_symbols = Parser.PARSING_TABLE[symbol][self._token_queue.get().token]

                    if next_symbols[0] != 'e':
                        # e는 엡실론으로 추가하지 않는다.
                        # 역순으로 stack에 추가한다.
                        for i in range(-1, -(len(next_symbols) + 1), -1):
                            stack.push(next_symbols[i])

                    log(f"queue : {self._token_queue}")
                    log(f"stack : {stack}")
                    log(f"work  : {symbol} -> {next_symbols}\n")
        except KeyError:
            for token in self._token_queue:
                pass

            # 가장 먼저 발견되는 symbol을 기준으로 에러를 판단한다.
            for symbol in stack:
                print(symbol)
                if symbol == '}':
                    # stack에 오른쪽 중괄호가 남을 때
                    raise NotMatchedBraceError
                if symbol == ';':
                    # stack에 세미 콜론이 남을 때
                    raise NoSemiColonError(self._source_code, used_token_stack.get())


class Token:
    """토큰에 대한 정보를 저장하는 클래스"""

    def __init__(self, token: str, token_string: str, line_number: int):
        """
        Args:
            token: 토큰
            token_string: 토큰에 대응하는 소스 코드 상의 문자열
            line_number: 토큰이 위치한 소스 코드 상의 줄 번호
        """
        self._token = token
        self._token_string = token_string
        self._line_number = line_number

    def __str__(self):
        return self._token

    @property
    def token(self) -> str:
        return self._token

    @property
    def token_string(self) -> str:
        return self._token_string

    @property
    def line_number(self) -> int:
        return self._line_number


class InvalidTokenError(Exception):
    """어휘 분석 중 소스 코드에 알 수 없는 토큰이 있을 때 발생하는 에러

    에러 발생 시 해석할 수 없는 토큰을 출력하고, 소스 코드에서 문제의 토큰이 있는 줄을 출력한다.
    """

    def __init__(self, source_code: str, invalid_token: str, line_number: int):
        """
        Args:
            source_code: 전체 소스 코드
            invalid_token: 인식되지 않는 토큰 문자열
            line_number: 인식되지 않는 토큰 문자열이 있는 줄 번호
        """
        self._source_code = source_code
        self._invalid_token = invalid_token
        self._line_number = line_number

    def __str__(self):
        result_string = f"Compile Error >> 소스 코드에 알 수 없는 토큰이 있습니다 : '{self._invalid_token}'\n"
        code_list = self._source_code.split("\n")
        result_string += f"(line {self._line_number}):{code_list[self._line_number - 1]}\n"

        # '^'로 표시
        text_until_invalid_token = f"(line {self._line_number}):{code_list[self._line_number - 1]}\n".split(self._invalid_token)
        for i in range(0, len(text_until_invalid_token[0])):
            result_string += " "
        result_string += "^"

        return result_string


class NotMatchedBraceError(Exception):
    """구문 분석 중 소스 코드에 중괄호가 맞지 않을 때 발생하는 에러

    에러 발생 시 중괄호가 맞지 않음을 알린다.
    """

    def __str__(self):
        return "Compile Error >> 소스 코드에 중괄호가 서로 매칭되지 않습니다"


class NoSemiColonError(Exception):
    """구문 분석 중 소스 코드에서 세미 콜론이 필요한 곳에 없을 때 발생하는 에러

    에러 발생 시 세미 콜론이 없음을 알리고, 소스 코드에서 필요한 부분을 표시한다.
    """

    def __init__(self, source_code: str, token_before: Token):
        """
        Args:
            source_code: 전체 소스 코드
            token_before: 세미 콜론이 있어야할 위치의 바로 앞 토큰
        """
        self._source_code = source_code
        self._token_before = token_before

    def __str__(self):
        result_string = "Compile Error >> 소스 코드의 다음 위치에 ';'가 필요합니다\n"
        code_list = self._source_code.split("\n")
        result_string += f"(line {self._token_before.line_number}):{code_list[self._token_before.line_number - 1]}\n"

        # '^'로 표시
        text_until_invalid_token = f"(line {self._token_before.line_number}):{code_list[self._token_before.line_number - 1]}\n".split(self._token_before.token_string)
        for i in range(0, len(text_until_invalid_token[0]) + len(self._token_before.token_string)):
            result_string += " "
        result_string += "^"

        return result_string


compiler = Compiler()
file_path = input()
compiler.compile(file_path)
