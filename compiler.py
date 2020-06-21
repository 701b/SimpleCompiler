"""
Compiler
--------
Team Name : 떡볶이 국물이 옷에 팀
Student Name : 문태의, 성아영
"""

import re
import pandas as pd
from filemanager import FileReader, FileWriter
from datastructure import Stack, Queue

TAG = "Compiler"


def log(content: str) -> None:
    """터미널에 로그를 찍기 위한 함수"""
    print(f"{TAG} >> {content}")


class Token:
    """토큰에 대한 정보를 저장하는 클래스"""

    def __init__(self, token: str, token_string: str, line_number: int, block_number: int):
        """
        Args:
            token: 토큰 종류
            token_string: 토큰에 대응하는 소스 코드 상의 문자열
            line_number: 토큰이 위치한 소스 코드 상의 줄 번호
            block_number: 토큰이 속한 가장 가까운 블록 번호
        """
        self._token = token
        self._token_string = token_string
        self._line_number = line_number
        self._block_number = block_number

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

    @property
    def block_number(self) -> int:
        return self._block_number


class TokenNode:
    """token을 tree에 넣기 위한 클래스"""

    def __init__(self, token: Token):
        self._token = token
        self._children = []
        self._current = 0

    def add_child(self, child: Token) -> None:
        """token을 노드 클래스에 넣는 메서드

        Args:
            child: 추가할 토큰
        """
        self._children.append(TokenNode(child))

    def get_child(self, index: int):
        """자식 노드를 반환하는 클래스

        Args:
            index: 가져올 자식이 담긴 list의 index

        Returns: 자식 노드 중 index번째 노드
        """
        if index >= len(self._children):
            return None

        return self._children[index]

    @property
    def token(self):
        return self._token

    @token.setter
    def token(self, token: Token):
        self._token = token

    def __iter__(self):
        return self

    def __next__(self):
        if self._current < len(self._children):
            self._current += 1
            return self._children[self._current - 1]
        else:
            raise StopIteration


class TreeStackNode:
    """순회를 위해 부모에 대한 상태를 저장하는 노드

    next_child 메서드를 통해 몇번 째 자식으로 접근해야하는지 알려준다.
    """

    def __init__(self, token_node: TokenNode):
        self._token_node = token_node
        self._child_index = 0

    def next_child(self):
        self._child_index += 1
        return self._child_index - 1

    @property
    def token_node(self) -> TokenNode:
        return self._token_node


class SyntaxTree:
    """코드의 구문을 저장한 트리

    구문 분석 단계의 결과로 생성된다.

    in 키워드로 반복문을 사용하면 preorder로 순회한다.
    트리 예시)
                                 total
                     prog                       $
        word   (  )        block

    순서 예시)
        total - prog - word - ( - ) - block - $
    """

    def __init__(self):
        self._root: TokenNode = None
        self._parent_stack = Stack()
        self._current: TreeStackNode = None

    @property
    def root(self) -> TokenNode:
        return self._root

    @root.setter
    def root(self, token_node: TokenNode):
        self._root = token_node

    def __iter__(self):
        return self

    def __next__(self) -> TokenNode:
        if self._current is None:
            if self._root is not None:
                self._current = TreeStackNode(self._root)
                return self._current.token_node
            else:
                raise StopIteration

        while True:
            child = self._current.next_child()

            if self._current.token_node.get_child(child) is not None:
                self._parent_stack.push(self._current)
                self._current = TreeStackNode(self._current.token_node.get_child(child))
                return self._current.token_node
            else:
                if not self._parent_stack.is_empty():
                    self._current = self._parent_stack.pop()
                    continue
                else:
                    self._current = None
                    raise StopIteration


class SymbolTable:
    """심볼 테이블을 담당하는 클래스

    add_symbol 메서드를 통해 추가할 수 있으며,
    추가된 기호들은 0x00000000부터 순서대로 메모리 주소가 할당된다.
    할당된 메모리는 search_address_of 메서드를 통해 찾을 수 있다.
    """

    BYTES_OF_INT = 4
    BYTES_OF_CHAR = 1

    def __init__(self, source_code: str):
        self._source_code = source_code
        self._symbol_table = pd.DataFrame({'identifier': [],
                                           'type': [],
                                           'block_number': [],
                                           'size': [],
                                           'address': []})
        self._next_address = 0x00000000

    def add_symbol(self, token: Token, type: str) -> None:
        """심볼 테이블에 심볼을 추가하는 메서드

        Args:
            token: 추가할 변수의 토큰
            type: 추가할 변수의 타입

        Raises:
            RedundantVariableDeclarationError: 심볼 테이블에 이미 같은 식별자, 블록 넘버의 심볼이 있을 때 발생한다.
            RuntimeError: int, char 이외의 변수가 입력되는 경우 발생한다.
        """
        search_result = self._symbol_table[(self._symbol_table['identifier'] == token.token_string) & (self._symbol_table['block_number'] == token.block_number)]
        if len(search_result) != 0:
            raise RedundantVariableDeclarationError(self._source_code, token)

        if type == 'int':
            if self._next_address % SymbolTable.BYTES_OF_INT != 0:
                self._next_address += SymbolTable.BYTES_OF_INT - self._next_address % SymbolTable.BYTES_OF_INT

            self._symbol_table.loc[len(self._symbol_table)] = [token.token_string, type, token.block_number, SymbolTable.BYTES_OF_INT, self._next_address]
            self._next_address += SymbolTable.BYTES_OF_INT
        elif type == 'char':
            self._symbol_table.loc[len(self._symbol_table)] = [token.token_string, type, token.block_number, SymbolTable.BYTES_OF_CHAR, self._next_address]
            self._next_address += SymbolTable.BYTES_OF_CHAR
        else:
            # 이 에러가 일어나면 코딩이 잘못된 것
            raise RuntimeError("Compiler >> Invalid type!")

    def search_address_of(self, token: Token, block_number_stack: Stack) -> int:
        """심볼 테이블에서 변수의 주소를 검색하는 메서드

        Args:
            token: 검색할 토큰
            block_number_stack: 해당 토큰이 속한 블록 스택

        Returns: 심볼 테이블에서 탐색한 변수의 주소

        Raises:
            NoVariableDeclarationError: 선언되지 않은 변수를 사용할 때 발생한다.
        """
        while True:
            block_number = block_number_stack.pop()
            search_result = self._symbol_table[self._symbol_table['identifier'] == token.token_string and self._symbol_table['block_number'] == block_number]

            if len(search_result) > 1:
                # 같은 범위에 같은 식별자의 변수가 중복됨.
                raise RuntimeError("Compiler >> redundant variable")
            elif len(search_result) == 1:
                # 찾은 경우
                return search_result[0][4]
            else:
                # 찾지 못한 경우
                if not block_number_stack.is_empty():
                    # 스택에 블록 넘버가 남은 경우
                    continue
                else:
                    # 스택이 빈 경우 -> 정의된 변수가 없음
                    raise NoVariableDeclarationError(self._source_code, token)


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
            scanner.scan()
        except FileNotFoundError:
            log(f"다음의 주소에서 파일을 읽지 못했습니다 : {file_path}")
            return
        except InvalidTokenError as e:
            print(e)
            return
        except NotMatchedBraceError as e:
            print(e)
            return

        parser = Parser(scanner.get_token_queue(), source_code)

        try:
            list = parser.parse()
            syntax_tree = list[0]
            symbol_table = list[1]
            print(symbol_table._symbol_table)
            Syntax_tree_Optimization(syntax_tree)
        except NoSemiColonError as e:
            print(e)
            return
        except NotDefinedCompileError as e:
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

    def scan(self) -> None:
        """어휘 분석을 시작하는 메서드

        소스 코드로부터 토큰을 추출한다.
        토큰은 Token클래스에 저장되어 token 리스트에 담긴다.

        Raises:
            FileNotFoundError: 인자로 받은 file_path로부터 파일을 읽지 못한 경우 발생한다.
            InvalidCodeError: 소스 코드에 알 수 없는 기호가 있을 때 발생한다.
            NotMatchedBraceError: 왼쪽과 오른쪽 중괄호 개수가 일치하지 않을 때 발생한다.
        """
        # block_number을 지정하기 위한 숫자
        # {를 만날 때 1이 증가된 블록 번호를 부여한다.
        # 전역 범위는 블록 번호가 0이다.
        block_number = 0

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
                    # {를 만날 때 블록 번호를 1 증가시킨다.
                    if result.group() == '{':
                        block_number += 1

                    log(f"{key}:{result.group()}")
                    token = Token(key, result.group(), line_number, block_number)
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

        # 왼쪽과 오른쪽 중괄호의 개수가 일치한지 확인한다.
        left_brace_count = 0
        right_brace_count = 0

        for token in self._tokens:
            if token.token == "{":
                left_brace_count += 1
            elif token.token == "}":
                right_brace_count += 1

        if left_brace_count != right_brace_count:
            raise NotMatchedBraceError()

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



def Syntax_tree_Optimization(syntax_tree):
    Code = []

    TreeStack = Stack()

    for token in syntax_tree:
        if token.token.token == "stat":
            TreeStack.push(token.token.token)

        if TreeStack.is_empty():
            continue
        elif TreeStack.get() == "stat":
            if token.token.token == "WHILE" or token.token.token == "IF" or token.token.token == "RETURN":
                Code.append(token.token.token)
                TreeStack.push(token.token.token)
            if token.token.token == "word":
                Code.append(token.token.token_string)
                TreeStack.push(token.token.token)

        elif TreeStack.get() == "WHILE":
            if token.token.token == "cond":
                Code.append(token.token.token)
                TreeStack.push(token.token.token)


        elif TreeStack.get() == "cond":
            if token.token.token != "block" and token.token.token != "THEN":
                if token.token.token_string is not None:
                    Code.append(token.token.token_string)
            if token.token.token == "block":
                TreeStack.pop()
                Code.append(token.token.token)
                TreeStack.push(token.token.token)
            if token.token.token == "THEN":
                TreeStack.pop()
                Code.append(token.token.token)
                TreeStack.push(token.token.token)

        elif TreeStack.get() == "block":
            if token.token.token == "}":
                TreeStack.pop()

        elif TreeStack.get() == "IF":
            if token.token.token == "cond":
                Code.append(token.token.token)
                TreeStack.push(token.token.token)

            if token.token.token == "ELSE":
                Code.append(token.token.token)
                TreeStack.push(token.token.token)

        elif TreeStack.get() == "THEN":
            if token.token.token == "block":
                TreeStack.pop()
                Code.append(token.token.token)
                TreeStack.push(token.token.token)

        elif TreeStack.get() == "ELSE":
            if token.token.token == "block":
                TreeStack.pop()
                Code.append(token.token.token)
                TreeStack.push(token.token.token)

        elif TreeStack.get() == 'word':
            if token.token.token == ";":
                TreeStack.pop()
                continue
            if token.token.token_string is not None:
                Code.append(token.token.token_string)

        elif TreeStack.get() == "RETURN":
            if token.token.token == ";":
                TreeStack.pop()
                continue
            if token.token.token_string is not None:
                Code.append(token.token.token_string)
    show_code(Code)

def show_code(Code):
    log(f"code : {Code}")



class Parser:
    """컴파일러의 구문 분석 단계를 담당하는 클래스

    파싱 테이블을 토대로 토큰들을 분석한다.
    """

    TOTAL_SYMBOL = "total"
    START_SYMBOL = "prog"
    END_SYMBOL = "$"

    PARSING_TABLE = {"prog": {"word": ["word", "(", ")", "block"],
                              "$": ["e"]},
                     "decls": {"word": ["e"],
                               "}": ["e"],
                               "IF": ["e"],
                               "WHILE": ["e"],
                               "RETURN": ["e"],
                               "int": ["decl", "decls"],
                               "char": ["decl", "decls"]},
                     "decl": {"int": ["vtype", "words", ";"],
                              "char": ["vtype", "words", ";"]},
                     "words": {"word": ["word", "words'"]},
                     "words'": {";": ["e"],
                                ",": [",", "word", "words'"]},
                     "vtype": {"int": ["int"],
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

    def parse(self) -> list:
        """Scanner로부터 받아온 Token들을 parse하는 메서드

        최초 기호와 마지막 기호가 담긴 stack과 Token들이 담긴 queue를 이용하여
        PARSING_TABLE에 따라 parse한다.
        LL parser의 원리를 사용한다.

        Returns: SyntaxTree, SymbolTable로 구성된 list를 반환한다.

        Raises:
            NoSemiColonError: 세미 콜론이 있어야할 위치에 없을 때 발생한다.
            NotDefinedCompileError: 에러를 파악할 수 없을 때 발생한다.
        """
        # Symbol Table
        symbol_table = SymbolTable(self._source_code)

        # 변수의 타입 기록
        vtype = None

        # 현재 word가 변수 선언할 때의 식별자인지 기록
        is_decl_var = False

        # Syntax Tree
        syntax_tree = SyntaxTree()

        # syntax tree에서 현재 접근 중인 노드의 스택 노드
        current_stack_node: TreeStackNode = None

        # syntax tree에서 현재 접근 중인 노드의 부모 스택
        parent_stack = Stack()

        # error handling을 위해 token이 담긴 queue에서 꺼낸 token들을 저장하는 stack
        used_token_stack = Stack()

        # symbol을 관리할 stack
        stack = Stack()

        stack.push(Parser.END_SYMBOL)
        stack.push(Parser.START_SYMBOL)

        root_node = TreeStackNode(TokenNode(Token(Parser.TOTAL_SYMBOL, None, None, 0)))
        root_node.token_node.add_child(Token(Parser.START_SYMBOL, None, None, 0))
        root_node.token_node.add_child(Token(Parser.END_SYMBOL, None, None, 0))
        syntax_tree.root = root_node.token_node
        parent_stack.push(root_node)
        current_stack_node = TreeStackNode(root_node.token_node.get_child(root_node.next_child()))

        self._token_queue.enqueue(Token(Parser.END_SYMBOL, "", 0, 0))

        log(f"queue : {self._token_queue}")
        log(f"stack : {stack}")
        log(f"work  : start\n")

        try:
            while not stack.is_empty():
                #for tree_stack_node in parent_stack:
                #    log(f"stack node : {tree_stack_node.token_node.token}")
                #for token_node in syntax_tree:
                #    log(f"node : {token_node.token} '{token_node.token.token_string}'")

                #log(f"current node : {current_stack_node.token_node.token}")

                if self._token_queue.get().token == stack.get():
                    # 기호가 같을 때 pop
                    token = self._token_queue.dequeue()
                    used_token_stack.push(token)
                    stack.pop()

                    # 토큰이 데이터 타입인 경우, 기록한다
                    if token.token == 'int' or token.token == 'char':
                        vtype = token.token
                    # 토큰이 세미 콜론인 경우, 심볼 테이블에 추가하는 것을 그만둔다.
                    elif token.token == ';':
                        is_decl_var = False
                    # 토큰이 word인 경우,
                    elif is_decl_var and token.token == 'word':
                        symbol_table.add_symbol(token, vtype)

                    # syntax tree의 현재 접근 중인 노드를 queue에서 빠진 token로 교체한다.
                    current_stack_node.token_node.token = token

                    # 부모로 올라가 다음 child로 이동한다.
                    if not token.token == Parser.END_SYMBOL:
                        while True:
                            current_stack_node = parent_stack.pop()
                            next_child = current_stack_node.token_node.get_child(current_stack_node.next_child())

                            if next_child is not None:
                                parent_stack.push(current_stack_node)
                                current_stack_node = TreeStackNode(next_child)
                                break

                    log(f"queue : {self._token_queue}")
                    log(f"stack : {stack}")
                    log(f"work  : pop\n")
                else:
                    # 기호가 다를 땐 확장
                    symbol = stack.pop()
                    next_symbols = Parser.PARSING_TABLE[symbol][self._token_queue.get().token]

                    # 기호가 decl인 경우, 변수 선언으로 간주하고 앞으로 나오는 word들을 심볼 테이블에 추가한다.
                    if symbol == 'decl':
                        is_decl_var = True

                    if next_symbols[0] != 'e':
                        # e는 엡실론으로 추가하지 않는다.
                        # 역순으로 stack에 추가한다.
                        for i in range(-1, -(len(next_symbols) + 1), -1):
                            stack.push(next_symbols[i])

                        # 원래 순서로 syntax tree의 현재 접근 중인 노드의 child로 추가한다.
                        for i in range(0, len(next_symbols)):
                            current_stack_node.token_node.add_child(Token(next_symbols[i], None, None, 0))

                        # 부모 스택에 현재 스택 노드를 추가한 후 첫번째 child로 이동
                        parent_stack.push(current_stack_node)
                        current_stack_node = TreeStackNode(
                            current_stack_node.token_node.get_child(current_stack_node.next_child()))
                    else:
                        # 부모로 올라가 다음 child로 이동한다.
                        while True:
                            current_stack_node = parent_stack.pop()
                            next_child = current_stack_node.token_node.get_child(current_stack_node.next_child())

                            if next_child is not None:
                                parent_stack.push(current_stack_node)
                                current_stack_node = TreeStackNode(next_child)
                                break

                    log(f"queue : {self._token_queue}")
                    log(f"stack : {stack}")
                    log(f"work  : {symbol} -> {next_symbols}\n")
        except KeyError:
            for token in self._token_queue:
                pass

            # stack에 남은 다음 token이 세미 콜론이면 세미 콜론이 있어야할 곳에 없는 것으로 판정한다.
            if stack.get() == ';':
                raise NoSemiColonError(self._source_code, used_token_stack.get())

            raise NotDefinedCompileError()

        return [syntax_tree, symbol_table]


class CodeGenerator:

    TARGET_FILE_NAME = "targetCode.txt"

    def __init__(self, code_table: list):
        """
        Args:
            code_table: op, arg1, arg2, result 순서로 저장된 코드 테이블
        """
        self._code_table = code_table

    def generate_code(self) -> None:
        """코드 테이블의 내용을 파일에 작성한다.

        Raises:
            RuntimeError: 코드 테이블에 정의되지 않은 instruction이 있는 경우 발생한다.
        """
        file_writer = FileWriter(CodeGenerator.TARGET_FILE_NAME)

        for intermediate_code in self._code_table:
            if intermediate_code[0] == "LD":
                # LD result, arg1 : arg1 주소에 있는 데이터를 result 레지스터에 저장한다.
                file_writer.write(f"{self._code_table[0]} {self._code_table[3]}, {self._code_table[1]}")
            elif intermediate_code[0] == "ST":
                # ST arg1, arg2 : arg1 레지스터에 있는 데이터를 arg2 주소의 메모리에 저장한다.
                file_writer.write(f"{self._code_table[0]} {self._code_table[1]}, {self._code_table[2]}")
            elif intermediate_code[0] == "LDB":
                # LDB result, arg1 : arg1 주소에 있는 byte단위 데이터를 result 레지스터에 저장한다.
                file_writer.write(f"{self._code_table[0]} {self._code_table[3]}, {self._code_table[1]}")
            elif intermediate_code[0] == "STB":
                # STB arg1, arg2 : arg1 레지스터에 있는 데이터를 arg2 주소의 메모리에 저장한다.
                file_writer.write(f"{self._code_table[0]} {self._code_table[1]}, {self._code_table[2]}")
            elif intermediate_code[0] == "ADD":
                # ADD result, arg1, arg2 : result = arg1 + arg2
                file_writer.write(f"{self._code_table[0]} {self._code_table[3]}, {self._code_table[1]}, {self._code_table[2]}")
            elif intermediate_code[0] == "MUL":
                # MUL result, arg1, arg2 : result = arg1 * arg2
                file_writer.write(f"{self._code_table[0]} {self._code_table[3]}, {self._code_table[1]}, {self._code_table[2]}")
            elif intermediate_code[0] == "LT":
                # LT result, arg1, arg2 : if (arg1 < arg2) result = 1 else result = 0
                file_writer.write(f"{self._code_table[0]} {self._code_table[3]}, {self._code_table[1]}, {self._code_table[2]}")
            elif intermediate_code[0] == "JUMPF":
                # JUMPF arg1   arg2 : if (arg1 == 0) Jump to arg2
                file_writer.write(f"{self._code_table[0]} {self._code_table[1]}   {self._code_table[2]}")
            elif intermediate_code[0] == "JUMPT":
                # JUMPT arg1   arg2 : if (arg1 != 0) Jump to arg2
                file_writer.write(f"{self._code_table[0]} {self._code_table[1]}   {self._code_table[2]}")
            elif intermediate_code[0] == "JUMP":
                # JUMPT    arg1 : Jump to arg1
                file_writer.write(f"{self._code_table[0]}    {self._code_table[1]}")
            else:
                # 정의되지 않은 instruction으로 코딩이 잘못된 경우임.
                raise RuntimeError("정의되지 않은 instruction 사용!")


class CompileError(Exception):
    """컴파일 에러를 다루는 클래스"""

    def __init__(self, source_code:str, error_token: Token, error_sentence = "", do_show_token = False, do_mark = False, do_mark_at_last = False):
        """
        Args:
            source_code: 전체 소스코드
            error_token:  에러가 발생한 토큰
            error_sentence: 발생한 에러에 대한 요약 문장
            do_show_token: 에러가 발생한 토큰 문자열을 보여줄 것인지
            do_mark: 소스 코드 상 에러가 발생한 토큰이 위치한 곳을 보여줄 것인지
            do_mark_at_last: 소스 코드 상 에러가 발생한 토큰이 위치한 곳의 뒷 부분을 마크할 것인지 / False이면 앞부분을 마크
        """
        self._source_code = source_code
        self._error_token = error_token
        self._error_setence = error_sentence
        self._do_show_token = do_show_token
        self._do_mark = do_mark
        self._do_mark_at_last = do_mark_at_last

    def __str__(self):
        result_string = f"Compile Error >> {self._error_setence}"

        if self._do_show_token:
            result_string += f" : {self._error_token.token_string}\n"
        else:
            result_string += "\n"

        if self._do_mark:
            code_list = self._source_code.split("\n")
            result_string += f"(line {self._error_token.line_number}):{code_list[self._error_token.line_number - 1]}\n"

            # '^'로 표시
            text_until_invalid_token = f"(line {self._error_token.line_number}):{code_list[self._error_token.line_number - 1]}\n".split(self._error_token.token_string)

            if self._do_mark_at_last:
                for i in range(0, len(text_until_invalid_token[0]) + len(self._error_token.token_string)):
                    result_string += " "
            else:
                for i in range(0, len(text_until_invalid_token[0])):
                    result_string += " "

            result_string += "^"

        return result_string


class InvalidTokenError(CompileError):
    """어휘 분석 중 소스 코드에 알 수 없는 토큰이 있을 때 발생하는 에러

    에러 발생 시 해석할 수 없는 토큰을 출력하고, 소스 코드에서 문제의 토큰이 있는 줄을 출력한다.
    """

    def __init__(self, source_code: str, error_token_str: str, line_number: int):
        super().__init__(source_code, Token(None, error_token_str, line_number), "소스 코드에 인식할 수 없는 토큰이 있습니다", True, True)


class NotMatchedBraceError(CompileError):
    """어휘 분석 중 소스 코드에서 왼쪽과 오른쪽 중괄호의 개수가 일치하지 않을 때 발생하는 에러

    에러 발생 시 왼쪽과 오른쪽 중괄호의 개수가 일치하지 않는다고 알린다.
    """

    def __init__(self):
        super().__init__(None, None, "소스 코드에서 왼쪽과 오른쪽 중괄호의 개수가 일치하지 않습니다.")


class RedundantVariableDeclarationError(CompileError):
    """기호표 작성 중 범위와 식별자가 중복된 변수 선언이 있을 때 발생하는 에러

    에러 발생 시 변수가 중복되었음을 알리고, 소스 코드에서 중복된 변수 위치를 표시한다.
    """

    def __init__(self, source_code: str, error_token: Token):
        super().__init__(source_code, error_token, "다음의 변수가 중복 선언되었습니다", True, True)


class NoSemiColonError(CompileError):
    """구문 분석 중 소스 코드에서 세미 콜론이 필요한 곳에 없을 때 발생하는 에러

    에러 발생 시 세미 콜론이 없음을 알리고, 소스 코드에서 세미 콜론이 필요한 부분을 표시한다.
    """

    def __init__(self, source_code: str, error_token: Token):
        super().__init__(source_code, error_token, "소스 코드의 다음 위치에 ';'가 필요합니다", False, True, True)


class NoVariableDeclarationError(CompileError):
    """심볼 테이블 사용 중 선언되지 않은 변수를 사용할 때 발생하는 에러

    에러 발생 시 선언되지 않은 변수가 사용되었음을 알리고, 선언되지 않은 변수가 사용된 위치를 표시한다.
    """

    def __init__(self, source_code: str, error_token: Token):
        super().__init__(source_code, error_token, "다음 변수가 선언되지 않았습니다", True, True)


class NotDefinedCompileError(CompileError):
    """컴파일 중 발생한 에러이지만, 원인을 알 수 없을 때 발생하는 에러"""

    def __init__(self):
        super().__init__(None, None, "컴파일 중 에러가 발생하였으나, 원인을 알 수 없습니다")


compiler = Compiler()
# file_path = input()
file_path = "reference/sample.c"
compiler.compile(file_path)
