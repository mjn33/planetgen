using System;

namespace SubdivisionPrototype {

public enum QuadPos {
    NONE,
    UPPER_LEFT,
    UPPER_RIGHT,
    LOWER_LEFT,
    LOWER_RIGHT,
}

public static class QuadPosMethods
{
    public static int to_idx(this QuadPos pos)
    {
        return (int)pos - 1;
    }
}

}

