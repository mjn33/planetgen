using System;

namespace SubdivisionPrototype {

public partial class MainWindow: Gtk.Window {
    public MainWindow()
        : base(Gtk.WindowType.Toplevel) {
        Build();
    }

    protected void OnDeleteEvent(object sender, Gtk.DeleteEventArgs a) {
        Gtk.Application.Quit();
        a.RetVal = true;
    }

    public void do_init() {
        this.Title = "Subdivision Test";

        MainCanvas a = new MainCanvas();

        Gtk.Box box = new Gtk.HBox (true, 0);
        box.Add(a);
        this.Add(box);
        this.Resize(500, 500);
        this.ShowAll();
    }
}

}
