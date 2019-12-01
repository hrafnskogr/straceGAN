#
import os
from response.requestHandler import RequestHandler


class StaticHandler(RequestHandler):
	def __init__(self):
		self.filetypes = {
			".apk": "binary",
			"notfound": "text/plain"
		}

	def find(self, file_path):
		split_path = os.path.splitext(file_path)
		extension = split_path[1]

		try:
			if(extension in (".apk")):
				self.contents = open(f"public{file_path}", 'rb')
			else:
				self.contents = open(f"public{file_path}", 'r')

			self.setContentType(extension)
			self.setStatus(200)
			return True
		except:
			self.setContentType('notfound')
			self.setStatus(404)
			return False

	def setContentType(self, ext):
		self.contentType = self.filetypes[ext]
