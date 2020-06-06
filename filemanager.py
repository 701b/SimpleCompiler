"""
FileManager
-----------
Team Name : 떡볶이 국물이 옷에 팀
Student Name : 문태의, 성아영
"""


class FileReader:
    """지정한 파일을 읽어 그 내용을 추출하는 클래스

    How to use
    ----------
    객체 생성 때 읽을 파일명과 파일 주소를 문자열 타입으로 file_path 인자에 넘긴다.
        file_reader = FileReader("sample.c")

    read_all 메소드로 파일의 내용을 문자열 타입으로 반환받는다.
        file_content = file_reader.read_all()
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path: 읽을 파일의 주소
        """
        self._file_path = file_path

    def read_all(self) -> str:
        result = ""

        with open(self._file_path, 'r') as file:
            for line in file:
                result += line

        return result


class FileWriter:
    """지정한 파일에 내용을 쓰게 해주는 클래스

    How to use
    ----------
    객체 생성 때 내용을 작성할 파일명과 파일 주소를 문자열 타입으로 file_path 인자에 넘긴다.
        file_reader = FileReader("sample.c")

    write 메소드에 쓸 내용을 물자열 타입으로 content 인자에 넘겨 파일에 내용을 작성할 수 있다. 여러번 호출하면 이어서 작성된다.
        file_content = file_reader.write("sample code")
    """

    def __init__(self, file_path: str):
        """
        Args:
            file_path: 쓰기 위해서 열 파일의 주소
        """
        self._file_path = file_path

    def write(self, content: str) -> None:
        with open(self._file_path, 'a') as file:
            file.write(content)
