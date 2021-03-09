var quillEditor = new Quill('#editor', {
    modules: {
        toolbar: [
            [{ header: [1, 2, false] }],
            ['bold', 'italic', 'underline', 'code', 'background', 'color']
        ]
    },
    placeholder: 'Write away ...',
    theme: 'snow'
});

var quillPreview = new Quill('#preview', {
    modules: {
        toolbar: false
    },
    readOnly: true,
    placeholder: 'No test to preview yet ...',
    theme: 'snow'
});


// function syncEditorPreview() {

// }


console.log(quillEditor.getContents())

quillEditor.on('text-change', function (delta, oldDelta, source) {
    if (source == 'api') {
        console.log("An API call triggered this change.");
    } else if (source == 'user') {
        console.log("A user action triggered this change.");
        quillPreview.updateContents(delta);
    }

});