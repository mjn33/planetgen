using System;
using Gtk;

namespace SubdivisionPrototype {

class MainClass {
    public static void Main(string[] args) {
        Application.Init();
        MainWindow win = new MainWindow();
        win.Show();
        win.do_init();
        Application.Run();
    }
}

}
