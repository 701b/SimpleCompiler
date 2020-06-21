main()
{
    int koreanScore, englishScore;
    char koreanGrade, englishGrade;

    int cutlineA, cutlineB, cutlineC, cutlineD;

    koreanScore = 80;
    englishScore = 65;

    cutlineA = 90;
    cutlineB = 80;
    cutlineC = 70;
    cutlineD = 60;

    IF koreanScore > cutlineA THEN
    {
        koreanGrade = 65;
    }
    ELSE
    {
        IF koreanScore > cutlineB THEN
        {
            koreanGrade = 66;
        }
        ELSE
        {
            IF koreanScore > cutlineC THEN
            {
                koreanGrade = 67;
            }
            ELSE
            {
                iF koreanScore > cutlineD THEN
                {
                    koreanGrade = 68;
                }
                ELSE
                {
                    koreanGrade = 70;
                }
            }
        }
    }

    IF englishScore > cutlineA THEN
    {
        englishGrade = 65;
    }
    ELSE
    {
        IF englishScore > cutlineB THEN
        {
            englishGrade = 66;
        }
        ELSE
        {
            IF englishScore > cutlineC THEN
            {
                englishGrade = 67;
            }
            ELSE
            {
                iF englishScore > cutlineD THEN
                {
                    englishGrade = 68;
                }
                ELSE
                {
                    englishGrade = 70;
                }
            }
        }
    }


}