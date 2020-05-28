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

    TAG = "compiler"

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


class Scanner:
    """컴파일러의 어휘 분석 단계를 수행하는 클래스

    메서드 analysis가 어휘 분석의 모든 단계를 수행하도록 한다.
    """

    TAG = "Scanner"

    REGULAR_EXPRESSION_OF_TOKEN = {'delimiter': '\(|\)|\{|\}|,|;',
                                   'keyword': '(IF)|(ELSE)|(THEN)|(WHILE)',
                                   'vtype': '(int)|(char)',
                                   'operator': '=|>|\+|\*',
                                   'word': '([a-z]|[A-Z])+',
                                   'num': '[0-9]+'}

    def __init__(self):
        self._source_code = ""
        self._tokens = []

    def scan(self, file_path: str) -> None:
        """어휘 분석을 시작하는 메서드

        총 세 단계로 나뉘며, 그 단계는 다음과 같다.
            * source code가 작성된 파일의 내용을 불러오는 단계
            * 불러온 내용을 전처리하는 단계
            * 전처리된 내용을 토큰으로 나누어 저장하는 단계

        Args:
            file_path: 분석할 소스 코드가 작성된 파일의 주소

        Raises:
            FileNotFoundError: 인자로 받은 file_path로부터 파일을 읽지 못한 경우 발생한다.
        """

        def read_source_code_from(file_path: str) -> None:
            """파일을 읽어온다."""
            file_reader = FileReader(file_path)
            self._source_code = file_reader.read_all()

        def preprocess() -> None:
            """코드 내용 중 공백, 줄바꿈을 모두 제거한다."""
            self._source_code = self._source_code.replace(' ', '')
            self._source_code = self._source_code.replace('\n', '')

        def split_to_token() -> None:
            """준비된 코드로부터 토큰을 추출한다."""
            source_code_temp = self._source_code

            while source_code_temp != "":
                for key, value in Scanner.REGULAR_EXPRESSION_OF_TOKEN.items():
                    pattern = re.compile(value)
                    result = pattern.match(source_code_temp)

                    if result is not None:
                        print(f"{key}:{result.group()}")
                        self._tokens.append(f"{key}:{result.group()}")
                        source_code_temp = source_code_temp.replace(result.group(), "", 1)
                        break

        log(Scanner.TAG, f"다음의 경로에서 소스 코드를 불러옵니다 : {file_path}")
        read_source_code_from(file_path)
        log(Scanner.TAG, f"읽어온 소스 코드::\n{self._source_code}\n")

        log(Scanner.TAG, "코드를 분석하기 전에 전처리 과정을 수행합니다")
        preprocess()
        log(Scanner.TAG, f"전처리된 소스 코드::\n{self._source_code}\n")

        log(Scanner.TAG, "준비된 코드를 토큰으로 분리합니다.")
        split_to_token()
        log(Scanner.TAG, f"토큰 목록::\n{self._tokens}\n")

    def get_token(self, index: int) -> str:
        return self._tokens[index]
