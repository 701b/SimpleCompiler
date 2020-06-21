main()
{
    int factorialNumber;
    int num;
    int result;

    factorialNumber = 100;
    num = 1;
    result = 1;

    WHILE factorialNumber + 1 > num
    {
        factorialNumber = result * num;
        num = num + 1;
    }

    RETURN result;
}