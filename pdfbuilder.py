import os
import fitz  # PyMuPDF

# Directory containing the PDFs
iList = 6
energy = .034
filename = "SmoothedState" + str(iList) + "_TileStart0_TileState0_Smooth" +str(energy)
outname = "Patch4Smoothed"
pdf_dir = filename


output_pdf = outname+'.pdf'

# Create a PDF writer object
pdf_writer = fitz.open()

# List of PDF files in the directory
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
def custom_key(item):
    first_period_index = item.find('.')
    second_period_index = item.find('.', first_period_index + 1)
    substring = item[:second_period_index]
    return float(substring)

pdf_files = sorted(pdf_files, key=custom_key)

pdf_writer = fitz.open()

# Loop through each PDF file
for pdf_file in pdf_files:
    pdf_path = os.path.join(pdf_dir, pdf_file)

    # Open the PDF and get the first page
    pdf_document = fitz.open(pdf_path)
    pdf_page = pdf_document[0]

    # Crop the whitespace
    rect = pdf_page.bound()
    pdf_page.set_cropbox(rect)

    # Append the page to the writer
    pdf_writer.insert_pdf(pdf_document)

    # Close the PDF document
    pdf_document.close()

# Save the combined and cropped PDF
pdf_writer.save(output_pdf)
pdf_writer.close()

print("Combined and cropped PDF created:", output_pdf)