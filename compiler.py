"""
Compiler
--------
Team Name : 떡볶이 국물이 옷에 팀
Student Name : 문태의, 성아영
"""

import re
from filemanager import FileReader


def log(tag: str, content: str) -> None:
    """터미널에 로그를 찍기 위한 함수"""
    print(f"{tag} >> {content}")


class Compiler:
    """컴파일러 클래스

    컴파일을 수행하기 위한 모든 클래스를 제어한다.

    compile 메서드를 통해 소스 파일의 코드를 타겟 코드로 컴파일한다.
    """

    TAG = "Compiler"

    def __init__(self):
        self._scanner = Scanner()

    def compile(self, file_path: str) -> None:
        """컴파일을 시작하는 메서드

        Args:
            file_path: 컴파일할 소스 코드가 작성된 파일의 주소
        """

        try:
            log(Compiler.TAG, "컴파일을 시작합니다.")
            self._scanner.scan(file_path)
        except FileNotFoundError:
            log(Compiler.TAG, f"다음의 주소에서 파일을 읽지 못했습니다 : {file_path}")
        except InvalidCodeError as e:
            print(e)


class Scanner:
    """컴파일러의 어휘 분석 단계를 수행하는 클래스

    메서드 analysis가 어휘 분석의 모든 단계를 수행하도록 한다.
    """

    TAG = "Scanner"

    REGULAR_EXPRESSION_OF_TOKEN = {'delimiter': '\(|\)|\{|\}|,|;',
                                   'keyword': '(IF)|(ELSE)|(THEN)|(WHILE)',
                                   'vtype': '(int)|(char)',
                                   'operator': '=|==|>|\+|\*',
                                   'word': '([a-z]|[A-Z])+',
                                   'num': '[0-9]+'}

    def __init__(self):
        self._source_code = ""
        self._tokens = []

    def scan(self, file_path: str) -> None:
        """어휘 분석을 시작하는 메서드

        총 세 단계로 나뉘며, 그 단계는 다음과 같다.
            * source code가 작성된 파일의 내용을 불러오는 단계
            * 불러온 source code를 토큰으로 나누어 저장하는 단계

        Args:
            file_path: 분석할 소스 코드가 작성된 파일의 주소

        Raises:
            FileNotFoundError: 인자로 받은 file_path로부터 파일을 읽지 못한 경우 발생한다.
            InvalidCodeError: 소스 코드에 알 수 없는 기호가 있을 때 발생한다.
        """

        def read_source_code_from(file_path: str) -> None:
            """파일을 읽어온다."""
            file_reader = FileReader(file_path)
            self._source_code = file_reader.read_all()

        def split_to_token() -> None:
            """준비된 코드로부터 토큰을 추출한다."""
            source_code_temp = self._source_code

            while source_code_temp != "":
                is_match = False

                for key, value in Scanner.REGULAR_EXPRESSION_OF_TOKEN.items():
                    pattern = re.compile(value)
                    result = pattern.match(source_code_temp)

                    if result is not None:
                        log(Scanner.TAG, f"{key}:{result.group()}")
                        self._tokens.append(f"{key}:{result.group()}")
                        source_code_temp = source_code_temp.replace(result.group(), "", 1)
                        is_match = True
                        break

                if is_match is False:
                    # 미리 정해진 정규 표현식에 패턴 매칭이 되지 않았다면, 공백과 문자열인지 확인한다.
                    pattern = re.compile("\s|\n")
                    result = pattern.match(source_code_temp)

                    if result is not None:
                        source_code_temp = source_code_temp.replace(result.group(), "", 1)
                    if result is None:
                        # 만약 공백과 문자열이 아니라면 에러를 발생시킨다.
                        pattern = re.compile("[^\s\n]*")
                        invalid_token = pattern.match(source_code_temp)

                        raise InvalidCodeError(self._source_code, invalid_token.group())

        log(Scanner.TAG, f"다음의 경로에서 소스 코드를 불러옵니다 : {file_path}")
        read_source_code_from(file_path)
        log(Scanner.TAG, f"읽어온 소스 코드::\n{self._source_code}\n")

        log(Scanner.TAG, "준비된 코드를 토큰으로 분리합니다.")
        split_to_token()
        log(Scanner.TAG, f"토큰 목록::\n{self._tokens}\n")

    def get_token(self, index: int) -> str:
        return self._tokens[index]


class InvalidCodeError(Exception):
    """소스 코드에 알 수 없는 토큰이 있을 때 발생하는 에러

    에러 발생 시 해석할 수 없는 토큰을 출력하고, 소스 코드에서 문제의 토큰이 있는 줄을 출력한다.
    """

    def __init__(self, source_code: str, invalid_token: str):
        """
        Args:
            source_code: 전체 소스 코드
            invalid_token: 인식되지 않는 토큰 문자열
        """
        self.source_code = source_code
        self.invalid_token = invalid_token

    def __str__(self):
        result_string = f"Compile Error >> 소스 코드에 알 수 없는 토큰이 있습니다 : '{self.invalid_token}'\n"
        code_list = self.source_code.split("\n")
        line_number = 1

        for code in code_list:
            temp = code.replace(self.invalid_token, "")

            if len(temp) != len(code):
                result_string += f"(line {line_number}) : {code}"
                break

            line_number += 1

        return result_string


compiler = Compiler()
file_path = input()
compiler.compile(file_path)